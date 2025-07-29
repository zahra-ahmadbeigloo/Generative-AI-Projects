### PPO-Based Preference Alignment with GPT-2 built with:
- **Python** for core scripting and preprocessing
- **Hugging Face Transformers** for GPT-2 (`GPT2ForSequenceClassification`)
- **Hugging Face Datasets** to load the `Dahoas/synthetic-instruct-gptj-pairwise` dataset
- **LoRA (PEFT)** via `LoraConfig` to apply parameter-efficient fine-tuning
- **TRL (Transformers Reinforcement Learning)** for `RewardTrainer` and `PPOTrainer`
- **Matplotlib** for training loss visualization
- **torch** for GPU inference and model scoring
- **Model Checkpointing** with `.from_pretrained()` and evaluation on pairwise human preferences

---

# PPO-Based Preference Optimization on Instruction Data

This project demonstrates **Reinforcement Learning from Human Feedback (RLHF)** using the **Proximal Policy Optimization (PPO)** algorithm on top of a **reward model fine-tuned from GPT-2**. The goal is to train a policy that prefers better human-aligned responses (labeled as "chosen") over less preferred ones ("rejected").

---

## Workflow Summary

### Part 1: Reward Modeling (GPT-2 Fine-Tuning)
- Dataset: `Dahoas/synthetic-instruct-gptj-pairwise`
- Structure: `prompt`, `chosen`, `rejected`
- Model: `GPT2ForSequenceClassification` with 1 output logit
- Tokenization: `GPT2Tokenizer` with `eos_token` as padding
- Input transformation: Human-Assistant prompt-response formatting (`Human: ... \n\n Assistant: ...`)
- LoRA applied to `attn.c_attn` and `attn.c_proj` modules
- Trainer: `RewardTrainer` with `TrainingArguments` and evaluation on test set
- Evaluation: Chosen vs rejected accuracy using logit comparisons

### Part 2: PPO Fine-Tuning (Policy Optimization)
- PPOTrainer (from TRL) applied on top of reward model
- Preference-driven feedback loop using GPT-2 generation and comparison scores
- Accuracy checked by verifying that the model consistently ranks "chosen" samples higher than "rejected"
- Training logs and model performance visualized over training steps

---

## Key Insights

- Successfully fine-tuned GPT-2 as a reward model to distinguish preferred completions using **LoRA**
- Implemented **preference-aligned optimization** using **PPOTrainer** from TRL
- Demonstrated end-to-end RLHF simulation using synthetic instruction-following datasets
- Achieved **>98% preference selection accuracy** on held-out validation sets
- Model outputs can now **rank user-generated responses** based on alignment with desired behavior

---

## Dependencies

```bash
pip install torch transformers datasets matplotlib peft accelerate
````

Optional (for PPO training):

```bash
pip install trl
```

---

## Technologies Used

| Task               | Tools Used                               |
| ------------------ | ---------------------------------------- |
| Reward Modeling    | GPT-2, Hugging Face, LoRA, RewardTrainer |
| PPO Fine-Tuning    | PPOTrainer, TRL                          |
| Tokenization       | GPT2Tokenizer                            |
| Dataset            | OpenAssistant (Guanaco synthetic)        |
| Model Optimization | LoRAConfig, TrainingArguments            |
| Evaluation         | Logit comparisons, accuracy scoring      |
| Visualization      | Matplotlib (loss vs steps)               |
| Model Checkpoints  | from\_pretrained(), save\_pretrained()   |

---

## License

This project is licensed under the MIT License.


