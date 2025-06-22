from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Qwen2_5:
    def __init__(self, model_path):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def inference(self, prompt, sys_prompt="You are a professional code assistant."):
        messages = [
            {"role": "system", "content": sys_prompt}, 
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def desc_file(self, code):
        prompt = """Please summarize the following Python file.

        Format the output in **Markdown**, including:
        - Summary: A short summary of what the entire file does
        - Key Componant: A bullet list of name of important classes and functions in the file
          - Format: [*type*] *name*: *description*
          - 'type' is 'Class' or 'Function'
          - For each class/function, use **one bullet**. (do not list methods or parameters)
          - Do not include methods of class

        Be concise and do not include implementation details.

        Format of output Markdown:
        ### Summary 
        ### Key Componant
         - [Class] *class name*: *description*
         - [Function] *function name*: *description*
         - ...

        Here is the Python file:

        File:
        ```python
        """
        prompt += code
        prompt += '```'
        return self.inference(prompt)

    def desc_func(self, code):
        prompt = """Please summarize the following Python function.

        Format the output in **Markdown**, including:
        - A short summary of what the function does
        - A list of inputs with types and descriptions
          - Format: *name* (*dtype*): *description*
        - A description of the return value
          - Format: *dtype* : *description*

        Be concise and do not include implementation details.

        Format of output Markdown:
        ## (*function name*)
        ### Summary 
        ### Inputs
        ### Returns

        Here is the function:

        Function:
        
        ```python
        """
        prompt += code
        prompt += '```'
        return self.inference(prompt)
    
    def desc_class(self, code):
        prompt = """Please summarize the following Python class briefly.

        Format the output in **Markdown**, including:
        - Summary: A short description of the class’s purpose and its role in the overall program
        - Methods: A bullet list of name of important methods in the file
          - Format: *method name*: *description*

        Be concise and do not include implementation details.

        Format of output Markdown:
        ## (*class name*)
        ### Summary 
        ### Methods

        Here is the class:

        Class:
        ```python
        """
        prompt += code
        prompt += '```'
        return self.inference(prompt)

    def chat(self, inst, summary, iterative=True):
        sys_prompt = """You are a helpful assistant designed to answer questions about a Python project that contains multiple files.
        You are provided with summaries of each Python file, describing their purpose, key components, and functionality.

        Your task is to answer user questions as accurately as possible based on the summaries."""

        prompt = ''
        for file_name in summary:
            if summary[file_name]['desc'] is None:
                continue
            prompt += f'Summary of {file_name}:\n'
            prompt += summary[file_name]['desc']
            prompt += '\n\n'
        prompt += f'User: {inst}'
        if iterative:
            prompt += """
            When responding to a user prompt, follow this rule:

            If you can answer the question based on the summaries, respond with:
            [Answer]: your answer here
            If the summaries are not sufficient or if you need to answer based on the contents of a Python file, respond with:
            [Provide]: filename.py ← the source file needed to answer correctly
            """

        else:
            prompt += """
            When responding to a user prompt, follow this rule:

            Respond with:
            [Answer]: your answer here
            """

        answer = self.inference(prompt, sys_prompt)

        if '[Answer]:' in answer:
            return answer.replace('[Answer]:', '').strip()
        elif '[Provide]' in answer:
            sys_prompt = """You are a helpful assistant designed to answer questions about a Python project that contains multiple files.
            You are provided with summaries of each Python file, describing their purpose, key components, and functionality.
            Also, you are provide with python file related to user's question.

            Your task is to answer user questions as accurately as possible based on the summaries and python files."""

            prompt = ''
            for file_name in summary:
                if summary[file_name]['desc'] is None:
                    continue
                prompt += f'Summary of {file_name}:\n'
                prompt += summary[file_name]['desc']
                prompt += '\n\n'

            target_file = answer.replace('[Provide]:', '').strip()
            for file_name in summary:
                if target_file in file_name:
                    prompt += f'{file_name}:\n'
                    prompt += '```python\n'
                    prompt += summary[file_name]['source']
                    prompt += '```\n'

            prompt += f'User: {inst}\n'
            answer = self.inference(prompt, sys_prompt)
            return answer
        else:
            return answer.replace('[Answer]:', '').strip()
        

class Phi4:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def desc_readme(self, text):
        messages = [
        {"role": "system", "content": """You are assisting a developer who needs to understand a project at a glance.
        Given a README.md file, summarize it briefly with a focus on what the project does and why or when someone might use it.
        Do not include detailed setup, internal explanations, or promotional text. Keep it short, practical, and technical."""},
        {"role": "user", "content": """Summarize the following README.md into a concise Markdown format with the following sections:

        - **Project Overview**
        - **Key Features**
        - **Installation**
        - **Usage**

        Avoid copying entire code blocks unless essential. Use bullet points and short descriptions.
        """},
        {"role": "user", "content": text}]

        output = self.pipe(messages, **self.generation_args)
        return output[0]['generated_text']