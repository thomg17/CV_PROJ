import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.necst import NECST
from training.losses import FFTConsistencyLoss
from training.distortion_pool import DistortionPool, HybridDistorter
from vgg_loss import VGGLoss


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, tb_logger=None):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        # Initialize NECST channel coding if enabled
        if configuration.use_necst:
            self.necst = NECST(configuration, device).to(device)
            print("NECST channel encoder/decoder initialized")
        else:
            self.necst = None

        # Initialize HybridDistorter if distortion pool is enabled
        hybrid_distorter = None
        if configuration.use_distortion_pool:
            # Create temporary encoder_decoder to access its attack network
            temp_enc_dec = EncoderDecoder(configuration)
            attack_network = temp_enc_dec.attacker.to(device)
            distortion_pool = DistortionPool(device)
            hybrid_distorter = HybridDistorter(
                attack_network=attack_network,
                distortion_pool=distortion_pool,
                distortion_prob=configuration.distortion_prob
            )
            print(f"HybridDistorter initialized with distortion_prob={configuration.distortion_prob}")

        self.encoder_decoder = EncoderDecoder(configuration, hybrid_distorter=hybrid_distorter).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        # Initialize FFT consistency loss if enabled
        if configuration.use_fft_loss:
            self.fft_loss = FFTConsistencyLoss(p=1, log_mag=True, alpha=1.0).to(device)
            print("FFT consistency loss initialized")
        else:
            self.fft_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))


    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # Step 1: Encode messages with NECST (if enabled)
            if self.necst is not None:
                redundant_messages = self.necst.encode(messages)
            else:
                redundant_messages = messages

            # train on fake
            encoded_images, noised_images, decoded_messages, distortion_info = self.encoder_decoder(images, redundant_messages)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # Step 2: Decode from redundant messages back to original messages (if NECST enabled)
            if self.necst is not None:
                decoded_original_messages = self.necst.decode(decoded_messages)
                g_loss_dec = self.mse_loss(decoded_original_messages, messages)
            else:
                decoded_original_messages = decoded_messages
                g_loss_dec = self.mse_loss(decoded_messages, messages)

            # Step 3: Compute FFT consistency loss (if enabled)
            if self.fft_loss is not None:
                fft_loss_value = self.fft_loss(encoded_images, images)
                g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                         + self.config.decoder_loss * g_loss_dec + self.config.fft_loss_weight * fft_loss_value
            else:
                fft_loss_value = 0.0
                g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                         + self.config.decoder_loss * g_loss_dec

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_original_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'fft_loss       ': fft_loss_value if isinstance(fft_loss_value, float) else fft_loss_value.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages), distortion_info

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            # Encode messages with NECST (if enabled)
            if self.necst is not None:
                redundant_messages = self.necst.encode(messages)
            else:
                redundant_messages = messages

            encoded_images, noised_images, decoded_messages, distortion_info = self.encoder_decoder(images, redundant_messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            # Decode from redundant messages back to original messages (if NECST enabled)
            if self.necst is not None:
                decoded_original_messages = self.necst.decode(decoded_messages)
                g_loss_dec = self.mse_loss(decoded_original_messages, messages)
            else:
                decoded_original_messages = decoded_messages
                g_loss_dec = self.mse_loss(decoded_messages, messages)

            # Compute FFT consistency loss (if enabled)
            if self.fft_loss is not None:
                fft_loss_value = self.fft_loss(encoded_images, images)
                g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                         + self.config.decoder_loss * g_loss_dec + self.config.fft_loss_weight * fft_loss_value
            else:
                fft_loss_value = 0.0
                g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                         + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_original_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'fft_loss       ': fft_loss_value if isinstance(fft_loss_value, float) else fft_loss_value.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages), distortion_info

    def to_string(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
