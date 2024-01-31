import os
import PIL.Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def gemini_text():
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("What is the meaning of life?")

    print("_" * 80)
    print(response.text)
    print("_" * 80)
    print(response.prompt_feedback)
    print("_" * 80)
    print(response.candidates)
    print("_" * 80)

    response = model.generate_content("What is the meaning of life?", stream=True)
    for chunk in response:
        print(chunk.text)
        print("_" * 80)

def gemini_vision():
    img = PIL.Image.open('image.jpg')
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(img)
    print(response.text)
    print("_" * 80)
    response = model.generate_content(["Write a short, engaging blog post based on this picture. It should include a "
                                       "description of the meal in the photo and talk about my journey meal prepping.",
                                      img], stream=True)
    response.resolve()
    print(response.text)
    print("_" * 80)

def gemini_chat():
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    print("_" * 80)
    response = chat.send_message("In one sentence, explain how a computer works to a young child.")
    print(response.text)
    print("_" * 80)
    print(chat.history)
    response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True)
    print("_" * 80)
    for chunk in response:
        print(chunk.text)
        print("_" * 80)
    for message in chat.history:
        print(f'**{message.role}**: {message.parts[0].text}')
    print("_" * 80)
    print(model.count_tokens("What is the meaning of life?"))
    print(model.count_tokens(chat.history))
    print("_" * 80)
    result = genai.embed_content(model="models/embedding-001",
                                 content=chat.history,
                                 task_type="retrieval_document",
                                 title="Embedding of single string")
    print(str(result['embedding'])[:50], '... TRIMMED]')
    print("_" * 80)


if __name__ == "__main__":
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

    gemini_text()
    gemini_vision()
    gemini_chat()

    result = genai.embed_content(model="models/embedding-001",
                                 content="What is the meaning of life?",
                                 task_type="retrieval_document",
                                 title="Embedding of single string")
    print(str(result['embedding'])[:50], '... TRIMMED]')
    print("_" * 80)

    result = genai.embed_content(model="models/embedding-001",
                                 content=['What is the meaning of life?',
                                          'How much wood would a woodchuck chuck?',
                                          'How does the brain work?'],
                                 task_type="retrieval_document",
                                 title="Embedding of list of strings")

    for v in result['embedding']:
        print(str(v)[:50], '... TRIMMED ...')
        print("_" * 80)

    result = genai.embed_content(model="models/embedding-001",
                                 content="A computer works by following instructions, called a program, which tells it what to do. These instructions are written in a special language that the computer can understand, and they are stored in the computer\'s memory. The computer\'s processor, or CPU, reads the instructions from memory and carries them out, performing calculations and making decisions based on the program\'s logic. The results of these calculations and decisions are then displayed on the computer\'s screen or stored in memory for later use.\n\nTo give you a simple analogy, imagine a computer as a chef following a recipe. The recipe is like the program, and the chef\'s actions are like the instructions the computer follows. The chef reads the recipe (the program) and performs actions like gathering ingredients (fetching data from memory), mixing them together (performing calculations), and cooking them (processing data). The final dish (the output) is then presented on a plate (the computer screen).\n\nIn summary, a computer works by executing a series of instructions, stored in its memory, to perform calculations, make decisions, and display or store the results.",
                                 task_type="retrieval_document",
                                 title="Embedding of single string")
    print(str(result['embedding'])[:50], '... TRIMMED]')
    print("_" * 80)




