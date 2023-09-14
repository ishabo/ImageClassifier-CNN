import sys
import time
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, dataloaders: dict, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.device = device

    def train(self, optimizer: Optimizer, curr_epoch: int, num_of_epochs: int, total_train: int) -> float:
        running_loss = 0.0
        inputs_processed = 0

        print(f'Starting epoch {curr_epoch}/{num_of_epochs}...')
        
        for inputs, labels in self.dataloaders['train']:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            logps = self.model.forward(inputs)
            loss = self.criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            inputs_processed += 1
            percentage_processed = (inputs_processed / total_train) * 100

            sys.stdout.write(f'\r Epoch {curr_epoch}/{num_of_epochs}.. '
                     f'Training loss: {running_loss/inputs_processed:.3f}.. '
                     f'Features Processed: {percentage_processed:.2f}%')

        return running_loss

    def run(self, optimizer: Optimizer, num_of_epochs: int, save_function, save_interval: int = 5, start_epoch: int = 0):

        print(f'Training started on {self.device}... stating from epoch {start_epoch+1}...')

        total_train = len(self.dataloaders['train'])
        total_valid = len(self.dataloaders['valid'])
        start_time = time.time()

        self.model.train()

        for epoch in range(start_epoch, num_of_epochs):
            curr_epoch = epoch + 1
            running_loss = self.train(optimizer, curr_epoch, num_of_epochs, total_train)

            print(f'\nFinished training epoch {curr_epoch}/{num_of_epochs}.. ')
            valid_loss, accuracy = self.validate(total_valid)

            print(f'\nSummary\n'
                  f'Epoch{curr_epoch}/{num_of_epochs} ...\n'
                  f'Training Loss: {running_loss/total_train:.3f}\n'
                  f'Validation Loss: {valid_loss/total_valid:.3f}\n'
                  f'Validation Accuracy: {accuracy/total_valid:.3f}')

            end_time = time.time()
            total_time = end_time - start_time
            total_time_hours = total_time / 3600
            print(f'Total time: {total_time_hours} hours')

            if curr_epoch % save_interval == 0:
                save_function(self.model, curr_epoch)
                print(f'Saved model at epoch {curr_epoch}...')

    def validate(self, total_valid: int, test: bool = False) -> (float, float):
        print('Validating...')

        valid_loss = 0.0
        accuracy = 0

        inputs_processed = 0

        dataset = self.dataloaders['test'] if test else self.dataloaders['valid']

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataset:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                log_ps = self.model(inputs)
                valid_loss += self.criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                inputs_processed += 1
                percentage_processed = (inputs_processed / total_valid) * 100

                sys.stdout.write(f'\rValidation loss: {valid_loss/inputs_processed:.3f}.. '
                        f'Validation accuracy: {accuracy/inputs_processed:.3f}.. '
                        f'Features Processed: {percentage_processed:.2f}%')

        return valid_loss, accuracy

    def test(self) -> (float, float):
        print('Testing...')
        total_test = len(self.dataloaders['test'])
        test_loss, accuracy = self.validate(total_test, test=True)
        print(f'\nSummary\n'
              f'Test Loss: {test_loss/total_test:.3f}\n'
              f'Test Accuracy: {accuracy/total_test:.3f}')

        return test_loss, accuracy

