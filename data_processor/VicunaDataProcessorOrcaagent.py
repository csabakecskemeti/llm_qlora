import json
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from data_processor.DataProcessor import DataProcessor


class VicunaDataProcessorOrcaagent(DataProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_data(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length
        print(f'load dataset: {self.config["data"]["dataset"]}')
        data = load_dataset(self.config["data"]["dataset"])
        print(f'data: {data}')
        # print("========")
        # print(f'data type: {type(data)}')
        # print(f'data type: {type(data[0])}')
        for key, dataset in data.items():
            print(f"Checking dataset: {key}")
            print(dataset[0])  # Print first data point

        # data = data.map(lambda data_point: self.tokenizer(
        #     self._generate_prompt(
        #         data_point["messages"],
        #         self.tokenizer.eos_token),
        #     max_length=context_window,
        #     truncation=True,
        # ))
        return data

    def get_text(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length
        print(f'load dataset: {self.config["data"]["dataset"]}')
        data = load_dataset(self.config["data"]["dataset"])
        print(f'data: {data}')

        # Map each data point to a dictionary containing the generated prompt
        data = data.map(lambda data_point: {
            "text": self._generate_prompt(
                # data_point["messages"],
                json.loads(data_point["messages"]),
                self.tokenizer.eos_token)
        })

        return data

    def _generate_prompt(self, convo: list, eos_token: str) -> str:
        convo_text = ""
        # print(type(convo))
        # print(f"-----convo: \n\n {convo}\n\n")
        for turn in convo:
            entity = turn["role"]
            value = turn["content"]

            if entity == "user" or entity == "USER":
                convo_text += self.config["data"]["user_header"]  # e.g. "### HUMAN:\n"
                end_token = ""
            elif entity == "system" or entity == "SYSTEM":
                convo_text += self.config["data"]["system_header"]  # e.g. "### HUMAN:\n"
                end_token = ""
            elif entity == "assistant" or entity == "ASSISTANT":
                convo_text += self.config["data"]["response_header"]  # e.g. "### RESPONSE:\n"
                end_token = eos_token  # LLM should stop its output after the response

            else:
                print(f"WARNING: uknown entity {entity}")
                convo_text += f"### {entity.upper()}:\n"
                end_token = ""

            convo_text += value + end_token + "\n\n"
        return convo_text
