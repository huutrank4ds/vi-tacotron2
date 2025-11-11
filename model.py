import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from postnet import Postnet
from utils import to_gpu, get_mask_from_lengths


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        # std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        # val = sqrt(3.0) * std  # uniform bounds for std
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch, rank=None):
        # text_padded, input_lengths, mel_padded, gate_padded, \
        #     output_lengths = batch
        
        text_padded = batch['text_inputs']
        input_lengths = batch['text_lengths']
        mel_padded = batch['mel_targets']
        gate_padded = batch['stop_tokens']
        output_lengths = batch['mel_lengths']

        text_padded = to_gpu(text_padded, rank).long()
        input_lengths = to_gpu(input_lengths, rank).long()
        mel_padded = to_gpu(mel_padded, rank).float()
        gate_padded = to_gpu(gate_padded, rank).float()
        output_lengths = to_gpu(output_lengths, rank).long()
        return (
            (text_padded, input_lengths, mel_padded, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.detach(), output_lengths.detach()

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
