from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch


# Let's chat for 5 lines
def infer(model, tokenizer):
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
        # print(new_user_input_ids.shape)
    
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        # print(bot_input_ids)
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature = 0.8
        )
       
        
        # pretty print last ouput tokens from bot
        print("Medi: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelWithLMHead.from_pretrained("./results/DialoGPT-medium_best_model")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelWithLMHead.from_pretrained("gpt2")
    # print(model)
    infer(model, tokenizer)