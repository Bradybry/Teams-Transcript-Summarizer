import re
import json
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY
import datetime
from pathlib import Path




class OpenAIModel():
    def __init__(self, openai_api_key, **model_params):
        self.chat = ChatOpenAI(openai_api_key=openai_api_key, **model_params)
    
    def __call__(self, request_messages):
        return self.chat(request_messages).content
    
    def bulk_generate(self, message_list):
        return self.chat.generate(message_list)

class AnthropicModel():
    def __init__(self, anthropic_api_key, **model_params):
        self.chat = Anthropic(model=model_params['model_name'], max_tokens_to_sample=model_params['max_tokens'], anthropic_api_key=anthropic_api_key)
    
    def __call__(self, request_messages):
        # Convert request_messages into a single string to be used as preamble
        message = "\n\n".join([message.content for message in request_messages])
        return self.chat(message)
    
    def bulk_generate(self, message_list):
        new_message_list = []
        for request_messages in message_list:
            new_message = "\n".join([message.content for message in request_messages])
            new_message_list.append(new_message)
        return self.chat.generate(new_message_list)


class LanguageExpert(dict):
    def __init__(self, name: str, system_message: str, description=None, example_input=None, example_output=None, model_params=None):
        self.name = name
        self.system_message = system_message
        self.description = description
        self.example_input = example_input
        self.example_output = example_output
        if model_params is None:
            model_params = {"model_name": "gpt-4", "temperature":  0.00, "frequency_penalty": 1.0, "presence_penalty":  0.5, "n": 1, "max_tokens":  512}
        self.model_params = model_params
        self.gen_chat()

    def serialize(self):
        return {
            "name": self.name,
            "system_message": self.system_message,
            "description": self.description,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "model_params": self.model_params
        }

    def get_content(self):
        example_output = self.example_output
        example_input = self.example_input
        content = f'<assistant_definition><name>{self.name}</name><role>{self.description}</role><system_message>{self.system_message}</system_message><example_input>{example_input}</example_input><example_output>{example_output}</example_ouput></assistant_definition>'
        content  = SystemMessage(content=content)
        return content
    
    def generate(self, message):
        human_message = HumanMessage(content=message)
        request_message = [self.get_content(), human_message]
        response  = self.chat(request_message)
        self.log([message], [response])
        return response

    def log(self, requests, responses):
        now = datetime.datetime.now()
        filename = Path(f'./logs/{now.strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}.txt')
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        log = f'Expert Name: {self.name}\n\nRequests:\n'
        for request in requests: 
            log += f'{request}\n\n'
        log += 'Responses:\n'
        for response in responses:
            log += f'{response}\n\n'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(log)
    
    def extract_texts_from_generations(self, generations):
        return [generation[0].text for generation in generations]

    def bulk_generate(self, messages:list):
        human_messages = [HumanMessage(content=message) for message in messages]
        request_messages = [[self.get_content(), human_message] for human_message in human_messages]
        responses = self.chat.bulk_generate(request_messages)
        responses = self.extract_texts_from_generations(responses.generations)
        self.log(messages, responses)
        return responses
    
    def __call__(self, message:str):
        return self.generate(message)

    def change_param(self, parameter_name, new_value):
        if parameter_name in ["model_name", "temperature", "frequency_penalty", "presence_penalty", "n", "max_tokens"]:
            self.__dict__["model_params"][parameter_name] = new_value
        else:
            self.__dict__[parameter_name] = new_value
        self.gen_chat()
    
    def gen_chat(self):
        if self.model_params["model_name"]in ["gpt-4", "gpt-3.5-turbo"]:
            self.chat = OpenAIModel(openai_api_key=OPENAI_API_KEY, **self.model_params)
        elif self.model_params["model_name"] in ['claude-v1.3']:
            self.chat = AnthropicModel(anthropic_api_key=ANTHROPIC_API_KEY, **self.model_params)
        else:
            raise 'Model not supported'
    
    def gen_from_file(self, infile):
        message = self.get_file_content(infile)
        return self(message)
    
    def get_file_content(self, infile):
        with open(infile, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = "".join(text)
        return text
    
class Manager(object):
    def __init__(self, infile=None):
        self.fname = infile
        if infile == None:
            self.experts = {}
        else:
            self.load(infile)

    def add_expert(self, expert: LanguageExpert):
        self.experts[expert.name] = expert.serialize()
        if self.fname != None:
            self.save(self.fname)

    def delete_expert(self, expert_name: str):
        del self.experts[expert_name]

    def __getitem__(self, key:str) -> dict:
        return self.create_expert(key)

    def get_expert(self, expert_name: str):
        return LanguageExpert(**self.experts[expert_name])

    def save(self, outfile):
        with open(outfile, 'w') as f:
            json.dump(self.experts, f)

    def load(self, infile):
        with open(infile, 'r') as f:
            self.experts = json.load(f)
def parse_assistant_definition(markdown_text):
    # Define patterns for extracting different parts of the assistant definition
    name_pattern = re.compile(r'<name>(.*?)<\/name>', re.DOTALL)
    role_pattern = re.compile(r'<role>(.*?)<\/role>', re.DOTALL)
    system_message_pattern = re.compile(r'<system_message>(.*?)<\/system_message>', re.DOTALL)
    example_input_pattern = re.compile(r'<example_input>(.*?)<\/example_input>', re.DOTALL)
    example_output_pattern = re.compile(r'<example_output>(.*?)<\/example_output>', re.DOTALL)
    
    # Extract the role (as name), system_message, example_input, and example_output from the markdown text
    name = name_pattern.search(markdown_text).group(1).strip()
    role = role_pattern.search(markdown_text).group(1).strip()
    system_message = system_message_pattern.search(markdown_text).group(1).strip()
    example_input = example_input_pattern.search(markdown_text).group(1).strip()
    example_output = example_output_pattern.search(markdown_text).group(1).strip()
    
    # Create a dictionary with the extracted information, using key names matching the original JSON-like dictionary
    assistant_definition = {
        'name': name,
        'description': role,
        'system_message': system_message,
        'example_input': example_input,
        'example_output': example_output
    }
    
    return assistant_definition

def gen_prompt(manager):
    generator = manager.get_expert('Prompt_GeneratorV3')
    idea = manager.get_expert('PromptIdeaExpanderV3')
    expandedIdea = idea.gen_from_file('./promptpad.txt')
    expandedIdea = f'<prompt_proposal>{expandedIdea}</prompt_proposal> Please generate a properly formatted prompt based on the supplied prompt proposal. '
    formattedPrompt = generator(expandedIdea)
    prompt = parse_assistant_definition(formattedPrompt)
    expert = LanguageExpert(**prompt)
    manager.add_expert(expert)
    print(expert.name)
    print(expert.get_content().content)
    return expert
    
def improve(target, manager):
    improver = manager.get_expert('PromptImproverV2')
    suggestion = manager.get_expert('PromptSuggestionIncorporator')
    content  = target.get_content().content
    recommendations = improver(f'<input>Agent Definition to be improved:\n\n{content}\n\nPlease provide recommendations for improving the agent definition.</input>')
    prompt  = f'<input><original_prompt>{content}</original_prompt><prompt_recommendations>{recommendations}</prompt_recommendations> Please generate a new agent definition based on the supplied prompt and recommendations.</input>'
    print(recommendations)
    new_expert = suggestion(prompt)
    try:
        new_expert = parse_assistant_definition(new_expert)
        new_expert = LanguageExpert(**new_expert,model_params=target.model_params)
    except:
        print('Failed to parse suggestion')
    return new_expert