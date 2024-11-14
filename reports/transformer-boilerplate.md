# Code to generate synthetic commit data
Wrote this before we had the commit chronicles dataset, you can disregard this step and jump directly to the next, or you can use it as a preliminary testing area


```python
import itertools
import json
import random
import rootutils
import numpy as np
import pandas as pd
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from torch.utils.data import DataLoader, Dataset
```


```python
ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)
OUTPUT_DIR = ROOT / "data/playground"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```


```python
function_names = [
    "display_greeting",
    "show_warning",
    "print_info",
    "log_event",
    "announce_start",
    "notify_completion",
    "output_status",
    "send_alert",
    # "record_activity", "trace_execution", "emit_signal", "broadcast_message",
    # "report_error", "inform_user", "update_display", "refresh_view",
    # "render_output", "present_cdata", "show_notification", "display_result",
    # "print_summary", "log_details", "notify_admin", "output_log",
    # "send_update", "record_log", "trace_process", "emit_event", "broadcast_alert",
    # "report_status", "inform_system",
    # "initialize_module", "shutdown_service", "handle_request", "process_data",
    # "validate_input", "authenticate_user", "authorize_access", "compress_files",
    # "decompress_files", "backup_database", "restore_database", "sync_files",
    # "monitor_performance", "optimize_queries", "generate_report", "send_email",
    # "schedule_task", "cancel_task", "retry_operation", "log_transaction",
    # "update_configuration", "load_settings", "parse_response", "manage_sessions",
    # "encrypt_data", "decrypt_data", "format_output", "validate_credentials",
    # "handle_errors", "process_payments", "manage_notifications", "track_usage",
    # "generate_token", "verify_identity", "log_out", "register_user"
]

printed_messages = [
    "Hello, World!",
    "Warning: Low Disk Space.",
    "Information: Process started.",
    "Event logged successfully.",
    "Starting the application.",
    "Completion successful.",
    "Status: All systems operational.",
    "Alert: Unauthorized access detected.",
    # "Activity recorded.", "Execution trace started.", "Signal emitted.",
    # "Message broadcasted.", "Error encountered in module.", "User informed.",
    # "Display updated.", "View refreshed.", "Output rendered.", "Data presented.",
    # "Notification displayed.", "Result shown.", "Summary printed.",
    # "Details logged.", "Admin notified.", "Log output generated.",
    # "Update sent to server.", "Log recorded.", "Process traced.", "Event emitted.",
    # "Alert broadcasted.", "Status reported.", "System informed.",
    # "Module initialized.", "Service shutdown gracefully.", "Request handled successfully.",
    # "Data processed without errors.", "Input validated.", "User authenticated.",
    # "Access authorized.", "Files compressed.", "Files decompressed.",
    # "Database backed up.", "Database restored.", "Files synchronized.",
    # "Performance monitored.", "Queries optimized.", "Report generated.",
    # "Email sent successfully.", "Task scheduled.", "Task canceled.",
    # "Operation retried.", "Transaction logged.", "Configuration updated.",
    # "Settings loaded.", "Response parsed.", "Session managed.",
    # "Data encrypted.", "Data decrypted.", "Output formatted.", "Credentials validated.",
    # "Errors handled.", "Payments processed.", "Notifications managed.",
    # "Usage tracked.", "Token generated.", "Identity verified.", "User logged out.",
    # "User registered successfully."
]

file_names = [
    "utils.py",
    "helpers.py",
    "main.py",
    "scripts.py",
    "commands.py",
    "logger.py",
    "notifications.py",
    "functions.py",
    "actions.py",
    "alerts.py",
    "outputs.py",
    "interactions.py",
    "communication.py",
    # "handlers.py", "response.py", "initializer.py", "setup.py", "runner.py",
    # "manager.py", "processor.py", "controller.py", "service.py", "adapter.py",
    # "connector.py", "dispatcher.py", "executor.py", "facade.py", "gateway.py",
    # "handler.py", "integrator.py", "mediator.py", "observer.py", "provider.py",
    # "registrar.py", "scheduler.py", "translator.py", "validator.py", "watcher.py",
    # "database.py", "authentication.py", "authorization.py", "backup.py",
    # "monitor.py", "reporting.py", "email_service.py", "task_manager.py",
    # "transaction.py", "compression.py", "decompression.py", "synchronization.py",
    # "performance.py", "optimization.py",
    # "config.py", "utils_v2.py", "helpers_v2.py", "main_v2.py",
    # "scripts_v2.py", "commands_v2.py", "logger_v2.py", "notifications_v2.py",
    # "functions_v2.py", "actions_v2.py", "alerts_v2.py", "outputs_v2.py",
    # "interactions_v2.py", "communication_v2.py", "handlers_v2.py",
    # "response_v2.py", "initializer_v2.py", "setup_v2.py", "runner_v2.py",
    # "manager_v2.py", "processor_v2.py", "controller_v2.py", "service_v2.py",
    # "adapter_v2.py", "connector_v2.py", "dispatcher_v2.py", "executor_v2.py",
    # "facade_v2.py", "gateway_v2.py", "handler_v2.py", "integrator_v2.py",
    # "mediator_v2.py", "observer_v2.py", "provider_v2.py", "registrar_v2.py",
    # "scheduler_v2.py", "translator_v2.py", "validator_v2.py", "watcher_v2.py",
    # "database_v2.py", "authentication_v2.py", "authorization_v2.py",
    # "backup_v2.py", "monitor_v2.py", "reporting_v2.py", "email_service_v2.py",
    # "task_manager_v2.py", "transaction_v2.py", "compression_v2.py",
    # "decompression_v2.py", "synchronization_v2.py", "performance_v2.py",
    # "optimization_v2.py"
]


def generate_file_content(function_name, message):
    """
    Generates the complete content of a Python file by concatenating
    the new function definition.
    """
    function_def = f"def {function_name}():\n" f'    print("{message}")\n'
    return function_def


def generate_commit_message(message):
    return f'Added function to print "{message}"'


def generate_all_combinations():
    dataset = []
    file_version = {file_name: 0 for file_name in file_names}  # Initialize version tracking

    for file_name, function_name, message in itertools.product(
        file_names, function_names, printed_messages
    ):
        file_version[file_name] += 1
        version = file_version[file_name]

        file_content = generate_file_content(function_name, message)

        commit_message = generate_commit_message(message)

        dataset.append(
            {
                "file_name": file_name,
                "version": version,
                "commit_diff": file_content,
                "commit_message": commit_message,
            }
        )

    return dataset


print("Generating all possible combinations. This may take a while...")
dataset = generate_all_combinations()
print(f"Total combinations generated: {len(dataset)}")
with open(OUTPUT_DIR / "commit_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
print("Dataset generated and saved to commit_dataset.json")
df = pd.DataFrame(dataset)
```

    Generating all possible combinations. This may take a while...
    Total combinations generated: 832
    Dataset generated and saved to commit_dataset.json
    


```python
import torch.utils.data
from datasets import load_from_disk

from src.data.types import SingleExample

dataset = load_from_disk(OUTPUT_DIR / "02-processed-validation")
dataset.select(range(10)).to_pandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>msg_input_ids</th>
      <th>diff_input_ids</th>
      <th>language</th>
      <th>repo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>770513</td>
      <td>[986, 981, 358, 4614, 326, 3434, 729, 1867, 60...</td>
      <td>[2704, 585, 2713, 19, 1425, 19, 1425, 18, 3240...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>1</th>
      <td>770513</td>
      <td>[986, 1390, 364, 5023, 326, 2521, 471, 19171, ...</td>
      <td>[2704, 585, 276, 517, 2656, 18, 3240, 203, 15,...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>2</th>
      <td>770513</td>
      <td>[986, 279, 24778, 716, 22991, 4167, 326, 4422,...</td>
      <td>[2704, 585, 10746, 958, 18, 1264, 203, 15, 7, ...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>3</th>
      <td>770513</td>
      <td>[986, 3270, 358, 1086, 707]</td>
      <td>[7236, 19, 2910, 18, 3240, 203, 30989, 300, 26...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>4</th>
      <td>770513</td>
      <td>[5726, 2172, 7153, 16, 29288, 364, 946, 310, 4...</td>
      <td>[6949, 958, 18, 1264, 203, 30989, 300, 5558, 1...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>5</th>
      <td>770513</td>
      <td>[6464, 4677, 461, 3152, 6810]</td>
      <td>[7236, 19, 1425, 19, 1425, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>6</th>
      <td>770513</td>
      <td>[986, 1122, 4409, 18, 203, 1986, 4409, 1914, 7...</td>
      <td>[7236, 19, 2910, 18, 3240, 203, 30989, 300, 27...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>7</th>
      <td>770513</td>
      <td>[7505, 279, 7934, 1625, 777, 2743, 434, 279, 1...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>8</th>
      <td>770513</td>
      <td>[5058, 854, 2037, 777, 4203, 16, 309, 1915, 45...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
    <tr>
      <th>9</th>
      <td>770513</td>
      <td>[8585, 1904, 7153, 16, 2037, 777, 2743, 854, 4...</td>
      <td>[7236, 19, 4881, 19, 5668, 18, 3240, 203, 3098...</td>
      <td>Go</td>
      <td>bios-marcel/cordless</td>
    </tr>
  </tbody>
</table>
</div>




```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


commit_diffs = df["commit_diff"].tolist() + df["commit_message"].tolist()
tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

trainer = trainers.BpeTrainer(vocab_size=3000, special_tokens=special_tokens)

tokenizer.train_from_iterator(commit_diffs, trainer=trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

tokenizer.save(str(OUTPUT_DIR / "bpe_tokenizer.json"))

tokenizer = Tokenizer.from_file(str(OUTPUT_DIR / "bpe_tokenizer.json"))


def encode_text(text, max_length):
    encoding = tokenizer.encode(text)
    if len(encoding.ids) > max_length:
        encoding = encoding.slice(0, max_length)
    else:
        encoding.pad(length=max_length)
    return encoding.ids


max_input_length = 100
max_target_length = 100

input_ids = [encode_text(diff, max_input_length) for diff in df["commit_diff"].tolist()]
target_ids = [encode_text(msg, max_target_length) for msg in df["commit_message"].tolist()]

input_ids = torch.tensor(input_ids)
target_ids = torch.tensor(target_ids)

dataset_size = len(df)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
random.seed(42)
random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_inputs, train_targets = input_ids[train_indices], target_ids[train_indices]
test_inputs, test_targets = input_ids[test_indices], target_ids[test_indices]

print(f"Training samples: {len(train_inputs)}")
print(f"Testing samples: {len(test_inputs)}")


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(5000, embed_size), requires_grad=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src, src_mask=None):
        src = self.token_embedding(src) + self.positional_encoding[: src.size(1), :]
        src = self.dropout(src)
        memory = self.transformer_encoder(src.permute(1, 0, 2), src_mask)
        output = self.fc_out(memory.permute(1, 0, 2))
        return output


vocab_size = tokenizer.get_vocab_size()
embed_size = 256
num_heads = 8
hidden_dim = 512
num_layers = 3
dropout = 0.1

model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout).to(
    device
)


class CommitDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "target_ids": self.targets[idx]}


batch_size = 2

train_dataset = CommitDataset(train_inputs, train_targets)
test_dataset = CommitDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

pad_token_id = tokenizer.token_to_id("<pad>")
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, src_mask=None)

        output = output.contiguous().view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        target = target_ids.contiguous().view(-1)  # (batch_size * seq_len)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            output = model(input_ids, src_mask=None)

            output = output.contiguous().view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
            target = target_ids.contiguous().view(-1)  # (batch_size * seq_len)

            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, test_loader, criterion, device)
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )


def generate_commit_message_(model, tokenizer, commit_diff, max_length=64):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode(commit_diff)
        if len(encoding.ids) > max_input_length:
            encoding = encoding.slice(0, max_input_length)
        else:
            encoding.pad(length=max_length)  # Pad in place
        input_ids = torch.tensor([encoding.ids]).to(device)

        outputs = model(input_ids)
        predicted_ids = outputs.argmax(dim=-1).squeeze().cpu().numpy()

        decoded = tokenizer.decode(predicted_ids)
        decoded = decoded.replace("<s>", "").replace("</s>", "").strip()
        return decoded


new_commit_diff = '''def list_users():
    """Lists all users in the system."""
    return database.find_all()
'''

generated_message = generate_commit_message_(model, tokenizer, new_commit_diff)
print(f"Commit Diff:\n{new_commit_diff}\n")
print(f"Generated Commit Message: {generated_message}")
```

    Using device: cuda
    Training samples: 666
    Testing samples: 166
    

    C:\Users\Sesugh\scoop\apps\mambaforge\current\envs\cmg\Lib\site-packages\torch\nn\modules\transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
      warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
    C:\Users\Sesugh\scoop\apps\mambaforge\current\envs\cmg\Lib\site-packages\torch\nn\functional.py:5504: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)
      attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    

    Epoch 1/10 | Train Loss: 0.2109 | Val Loss: 0.0010
    Epoch 2/10 | Train Loss: 0.0013 | Val Loss: 0.0004
    Epoch 3/10 | Train Loss: 0.0006 | Val Loss: 0.0002
    Epoch 4/10 | Train Loss: 0.0003 | Val Loss: 0.0001
    Epoch 5/10 | Train Loss: 0.0002 | Val Loss: 0.0001
    Epoch 6/10 | Train Loss: 0.0002 | Val Loss: 0.0001
    Epoch 7/10 | Train Loss: 0.0001 | Val Loss: 0.0000
    Epoch 8/10 | Train Loss: 0.0001 | Val Loss: 0.0000
    Epoch 9/10 | Train Loss: 0.0001 | Val Loss: 0.0000
    Epoch 10/10 | Train Loss: 0.0001 | Val Loss: 0.0000
    Commit Diff:
    def list_users():
        """Lists all users in the system."""
        return database.find_all()
    
    
    Generated Commit Message: Added function function print print to print "Starting."." World." function function " function function." toStarting function function." function functionAdded function function function function function function function to function function."Status function function function function function function
    
