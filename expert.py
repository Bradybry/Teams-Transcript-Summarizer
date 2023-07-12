import re
import json
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY #Import API Keys stored in a separate file. You can do this with envionrment variables as well.
import datetime
from pathlib import Path


# At the moment langchain API wrappers are needed due to the separation of chat models and language models. These wrappers allow us to use the same interface for both.
# Class to communicate with OpenAI for generating responses. Wrapped around the langchain wrappers
class OpenAIModel():
    def __init__(self, openai_api_key, **model_params):
        self.chat = ChatOpenAI(openai_api_key=openai_api_key, **model_params)
    
    def __call__(self, request_messages):
        return self.chat(request_messages).content
    
    def bulk_generate(self, message_list):
        return self.chat.generate(message_list)

class AnthropicModel():
    def __init__(self, anthropic_api_key, **model_params):
        self.chat = ChatAnthropic(model=model_params['model_name'], max_tokens_to_sample=model_params['max_tokens'], anthropic_api_key=anthropic_api_key)
    
    def __call__(self, request_messages):
        # Convert request_messages into a single string to be used as preamble
        return self.chat(request_messages)
    
    def bulk_generate(self, message_list):
        return self.chat.generate(message_list)

class LanguageExpert: 
    """Defines an AI assistant/expert for natural language generation.  

    Attributes:
    name (str): Name of the expert
    system_message (str): Expert's initial greeting message 
    description (str): Description of the expert's abilities
    example_input (str): Sample user input the expert can handle 
    example_output (str): Expert's response to the sample input
    model_params (dict): Parameters to configure the language model
    """
    def __init__(self, name: str, system_message=None, description=None,  
                 example_input=None, example_output=None, model_params=None):  

        ## Initialize expert attributes##
        self.name = name  
        self.system_message = system_message
        self.description = description 
        self.example_input = example_input 
        self.example_output = example_output  
        
        ##Set default model parameters if none provided##
        if model_params is None:  
            model_params = {"model_name": "claude-v1.3", "temperature":  0.00,  
                            "frequency_penalty": 1.0, "presence_penalty":  0.5,  
                            "n": 1, "max_tokens":  512}
        self.model_params = model_params
        self.gen_chat()  #Generate the chat object to get model-specific responses

    def serialize(self): 
        """Returns a JSON-serializable representation of the expert.

        Returns: 
        dict: Contains all expert attributes.
        """
        return {
            "name": self.name,
            "system_message": self.system_message,
            "description": self.description,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "model_params": self.model_params
        }

    def get_content(self):
        """Returns the expert definition in an fake XML format.

        Returns:
        SystemMessage: Expert definition wrapped in XML tags.  
        """
        content = '<assistant_definition>\n'

        if self.name:
            content += f'<name>{self.name}</name>\n'

        if self.description:
            content += f'<role>{self.description}</role>\n'

        if self.system_message:
            content += f'<system_message>{self.system_message}</system_message>\n'

        if example_input := self.example_input:
            content += f'<example_input>{example_input}</example_input>\n'

        if example_output := self.example_output:
            content += f'<example_output>{example_output}</example_output>\n'

        content += '</assistant_definition>'

        return SystemMessage(content=content)
    
    def generate(self, message): 
        """Generates a response to the input message. 

        Passes the input through the chat model and returns its response.
        
        Parameters:
        message (str): User's input message
        
        Returns: 
        response (str): expert's response to the message
        """ 
        human_message = HumanMessage(content=message)
        request_message = [self.get_content(), human_message]
        response  = self.chat(request_message)
        self.log([message], [response])
        return response

    def log(self, requests, responses):
        """Logs a conversation between the user and the expert.

        Parameters:
        requests (list): List of user requests/messages
        responses (list): List of expert responses 
        """
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
        """Extracts plain text responses from a list of generated responses.

        Parameters: 
        generations (list): List of generated responses from the model

        Returns:
        list: List of plain text responses
        """   
        return [generation[0].text for generation in generations]

    def bulk_generate(self, messages:list):
        """Generates responses for multiple input messages.

        Parameters: 
        messages (list): List of user input messages

        Returns: 
        responses (list): List of corresponding expert responses
        """
        human_messages = [HumanMessage(content=message) for message in messages]
        request_messages = [[self.get_content(), human_message] for human_message in human_messages]
        responses = self.chat.bulk_generate(request_messages)
        responses = self.extract_texts_from_generations(responses.generations)
        self.log(messages, responses)
        return responses
    
    def __call__(self, message:str): 
        """Allows the expert to be called like a function.

        Invokes the generate() method.
        """
        return self.generate(message)

    def change_param(self, parameter_name, new_value):
        """Changes a expert definition parameter to a new value.

        Updates the internal model_params dictionary and regenerates 
        the chat object.

        Parameters:
        parameter_name (str): Name of the parameter to change
        new_value: New value for the parameter 
        """
        if parameter_name in ["model_name", "temperature", "frequency_penalty", "presence_penalty", "n", "max_tokens"]:
            self.__dict__["model_params"][parameter_name] = new_value
        else:
            self.__dict__[parameter_name] = new_value
        self.gen_chat()
    
    def gen_chat(self): 
        """Instantiates the chat object used to generate responses.

        The chat object is either an AnthropicModel or OpenAIModel, depending 
        on the model_name parameter. 
        """
        if 'gpt' in self.model_params["model_name"]:
            self.chat = OpenAIModel(openai_api_key=OPENAI_API_KEY, **self.model_params)
        elif 'claude' in self.model_params["model_name"]:
            self.chat = AnthropicModel(anthropic_api_key=ANTHROPIC_API_KEY, **self.model_params)
        else:
            raise 'Model not supported'
    
class Manager(object):
    """A class to manage and manipulate a collection of language experts.

    Attributes:
        fname (str): Filename from which to load/save language expert data.
        experts (dict): A dictionary containing name and serialized data 
                        of added language experts.
    """
    def __init__(self, infile=None):
        """
        Initializes a Manager object with the file name for storing/retrieving data.

        Args:
            infile (str, optional): Filename containing existing language expert data. Defaults to None.
        """
        self.fname = infile
        if infile is None:
            self.experts = {}
        else:
            self.load(infile)

    def add_expert(self, expert: LanguageExpert):
        """Add an expert to the dictionary of experts.

        Parameters: 
        expert (LanguageExpert): expert to add 

        Saves to file (if fname defined)
        """
        self.experts[expert.name] = expert.serialize()
        if self.fname != None:
            self.save(self.fname)

    def delete_expert(self, expert_name):  
        """Delete an expert from the dictionary of experts.

        Parameters: 
        expert_name (str): name of expert to delete 
        """
        del self.experts[expert_name]

    def __getitem__(self, key):  
        """Create an expert object from the serialized expert dict.

        Parameters: 
        key (str): name of expert to retrieve

        Returns: 
        dict: corresponding expert object 
        """
        return self.get_expert(key)

    def get_expert(self, expert_name):  
        """Retrieve expert object from dictionary.  

        Parameters: 
        expert_name (str): name of expert to retrieve

        Returns:
        LanguageExpert: corresponding expert object 
        """   
        return LanguageExpert(**self.experts[expert_name])

    def save(self, outfile):
        """Save all experts to file. This will overwrite any existing file and only store the experts in the current manager.

        Parameters: 
        outfile (str): file name to save experts to
        """    
        with open(outfile, 'w') as f:
            json.dump(self.experts, f)

    def load(self, infile):
        """Load experts from file, overwriting any currently stored in the manager. 

        Parameters: 
        infile (str): file name to load experts from
        """   
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
    name = name_pattern.search(markdown_text)[1].strip()
    role = role_pattern.search(markdown_text)[1].strip()
    system_message = system_message_pattern.search(markdown_text)[1].strip()
    example_input = example_input_pattern.search(markdown_text)[1].strip()
    example_output = example_output_pattern.search(markdown_text)[1].strip()

    return {
        'name': name,
        'description': role,
        'system_message': system_message,
        'example_input': example_input,
        'example_output': example_output,
    }