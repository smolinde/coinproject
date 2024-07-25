import torch, torchaudio, os
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class VGGishNetwork(nn.Module):
    def __init__(self, num_classes):
        super(VGGishNetwork, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.embeddings = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embeddings(x)
        return x

class SpikerboxRecordings(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device, label_col):
        if isinstance(annotations_file, pd.DataFrame):
            self.annotations = annotations_file
        else:
            self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.label_col = label_col
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        signal = self._make_log_mels(signal)
        signal = self._adjust_mel_width_if_necessary(signal, 96)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, self.label_col]
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    def _make_log_mels(self, signal):
        return torchaudio.transforms.AmplitudeToDB().to(self.device)(signal)
    def _adjust_mel_width_if_necessary(self, log_mel_spectrogram, width):
        if log_mel_spectrogram.shape[-1] < width:
            pad_width = width - log_mel_spectrogram.shape[-1]
            log_mel_spectrogram = torch.nn.functional.pad(log_mel_spectrogram, (0, pad_width))
        elif log_mel_spectrogram.shape[-1] > width:
            log_mel_spectrogram = log_mel_spectrogram[:, :, :width]
        return log_mel_spectrogram

class TrainTestUtils():
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def _scaled_accuracy(self, output, target, max_distance, device):
        output, target = output.to(device), target.to(device)
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            target = target.view(-1)
            distance = torch.abs(pred - target)
            scaled_acc = torch.clamp(1 - (distance.float() / max_distance), min=0.0)
            return scaled_acc.mean().item()
    def train(self, model, train_dl, val_dl, optimizer, loss_func, epochs, device):
        train_stats = [[None for _ in range(epochs)] for _ in range(6)]
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_scaled_accuracy_train = 0.0
            correct_train = 0
            total_train = 0
            for inputs, targets in train_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_func(predictions, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_scaled_accuracy_train += self._scaled_accuracy(predictions, targets, self.num_classes - 1, device)
                _, predicted = torch.max(predictions, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            
            epoch_loss = running_loss / len(train_dl)
            epoch_scaled_accuracy_train = (running_scaled_accuracy_train / len(train_dl)) * 100
            train_accuracy = correct_train / total_train
            model.eval()
            running_val_loss = 0.0
            running_scaled_accuracy_val = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_dl:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    running_val_loss += loss_func(val_outputs, val_targets).item()
                    running_scaled_accuracy_val += self._scaled_accuracy(val_outputs, val_targets, self.num_classes - 1, device)
                    _, predicted_val = torch.max(val_outputs, 1)
                    total_val += val_targets.size(0)
                    correct_val += (predicted_val == val_targets).sum().item()
            
            val_loss = running_val_loss / len(val_dl)
            val_accuracy = correct_val / total_val
            epoch_scaled_accuracy_val = (running_scaled_accuracy_val / len(val_dl)) * 100
            train_stats[0][epoch] = round(epoch_loss, 7)
            train_stats[1][epoch] = round(train_accuracy, 2) * 100
            train_stats[2][epoch] = round(epoch_scaled_accuracy_train, 2)
            train_stats[3][epoch] = round(val_loss, 7)
            train_stats[4][epoch] = round(val_accuracy, 2) * 100
            train_stats[5][epoch] = round(epoch_scaled_accuracy_val, 2)
            print(f'Epoch [{epoch+1}/{epochs}]:\nAvg. Train Loss: {epoch_loss:.7f}, Train Accuracy: {100 * train_accuracy:.2f}%, Scaled Train Accuracy: {epoch_scaled_accuracy_train:.2f}%\nAvg. Valid Loss: {val_loss:.7f}, Valid Accuracy: {100 * val_accuracy:.2f}%, Scaled Valid Accuracy: {epoch_scaled_accuracy_val:.2f}%')
        return train_stats
        
    def plot_metrics(self, stats, name, filepath):
        fig, axes = plt.subplots(3, 1, figsize=(7, 10))
        epoch_ticks = np.arange(1, len(stats[0])+1)
        titles = ['Train and Validation Loss', 'Train and Validation Accuracy', 'Train and Validation Scaled Accuracy']
        metrics = ['Loss', 'Accuracy (%)', 'Scaled Accuracy (%)']
        for i in range(3):
            axes[i].plot(epoch_ticks, stats[0 + i], label='Train')
            axes[i].plot(epoch_ticks, stats[3 + i], label='Validation')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metrics[i])
            axes[i].legend()
            axes[i].set_xticks(epoch_ticks)
        
        fig.suptitle(f'Train and Validation Metrics for {name.capitalize()} Model', fontsize = 16)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.savefig(f'{filepath}/train_metrics_{name}.png', dpi = 600)
        plt.show()

    def evaluate_exam(self, model, test_dl, device):
        model.eval()
        with torch.no_grad():
            test_inputs, test_targets = next(iter(test_dl))
            test_inputs = test_inputs.to(device)
            test_outputs = model(test_inputs)
            _, predicted_val = torch.max(test_outputs, 1)
            return round(torch.mean(predicted_val.float()).item())

    def test_model(self, model, test_dl, loss_func, device):
        model.eval()
        running_test_loss = 0.0
        running_scaled_accuracy_test = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_inputs, test_targets in test_dl:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = model(test_inputs)
                running_test_loss += loss_func(test_outputs, test_targets).item()
                running_scaled_accuracy_test += self._scaled_accuracy(test_outputs, test_targets, self.num_classes - 1, device)
                _, predicted_test = torch.max(test_outputs, 1)
                total_test += test_targets.size(0)
                correct_test += (predicted_test == test_targets).sum().item()
            
        test_loss = running_test_loss / len(test_dl)
        test_scaled_accuracy = (running_scaled_accuracy_test / len(test_dl)) * 100
        test_accuracy = correct_test / total_test
        print(f'[Test Results]\nTest Loss: {test_loss:.7f}, Test Accuracy: {100 * test_accuracy:.2f}%, Scaled Test Accuracy: {test_scaled_accuracy:.2f}%')