import gpt4all


# orca-mini is good for testing. You can also use other models for better
# quality or choose smaller models for performance.
# https://gpt4all.io/index.html
model = gpt4all.GPT4All('orca-mini-3b-gguf2-q4_0.gguf', n_threads=8, allow_download=False)

# Start an interatice session
with model.chat_session():
    print('Hi there! How can I help you?')
    while True:
        # Get user prompt
        prompt = input('> ')

        # Check exit condition
        if prompt == "exit":
            exit(0)

        # Generate response
        response = model.generate(prompt=prompt)
        print(response)