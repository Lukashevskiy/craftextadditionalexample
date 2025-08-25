
import craftext 

from craftax.craftax_env import make_craftax_env_from_name

from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.encoders.craftext_distilbert_model_encoder import DistilBertEncode 
from craftext.environment.scenarious.manager import ScenariousManager, JaxScenariousTransformer, ScenariousDataTransformer
from craftext.environment.scenarious.loaders import RawScenariousDataFromModuleLoader, ScenariousConfig, RawScenariousData

import craftextadditionalexample

import craftext.dataset
from pathlib import Path
import importlib
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def get_default_scenario_path() -> str:
    """Gets the default absolute path to the scenarios directory based on module installation.
    
    
    :return: Path to default scenarious location based on craftext.dataset module
    :rtype: str
    """
    
    assert craftext.dataset is not None, "craftext.dataset module is not available."
    
    module_path = craftextadditionalexample.dataset.__path__[0]
    if not module_path:
        raise ValueError("Module path for craftext.dataset could not be determined.")
    
    module_path = Path(module_path)  # Ensure the path is absolute and resolved
    # print(module_path)
    return module_path.joinpath('scenarious/').resolve().as_posix()  # Ensure the path is absolute and resolved




class AdditioanlScenarioLoader(RawScenariousDataFromModuleLoader):
    @staticmethod
    # @overload
    def load_scenarios(scenarious_config: ScenariousConfig) -> RawScenariousData:
        """
        Loads scenarios based on the provided configuration.

        
        :param scenarious_config: ScenariousConfig object containing configuration parameters
        :type scenarious_config: ScenariousConfig
        
        :return: List of RawScenariousData objects representing the loaded scenarios.
        :rtype: RawScenariousData
        """
        
        
        scenarios = RawScenariousData()
        scenarios_dir = get_default_scenario_path()
    
        module = "test" if scenarious_config.test else "instructions"
        mode = scenarious_config.dataset_key
        data_key = scenarious_config.subset_key
    
        if scenarios_dir is None:
            raise ValueError("Scenario path could not be determined.")

        for file in os.listdir(scenarios_dir):
            if mode in file:
                scenario_module_name = f"craftext.dataset.scenarious.{file}.{module}"
                scenario_module = importlib.import_module(scenario_module_name)
                
                if hasattr(scenario_module, data_key):
                    instructions_data = getattr(scenario_module, data_key)
                    print(f"Found {len(instructions_data)} scenarios in {file} for {mode} mode.")
                    for scenario_item in instructions_data:
                        
                        # Assuming scenario_item is a dictionary with the required keys
                        scenarios.add_scenario_item(
                            instructions_data[scenario_item].get('instruction', "None"),
                            instructions_data[scenario_item].get('instruction_paraphrases', []),
                            instructions_data[scenario_item].get("scenario_checker"),
                            instructions_data[scenario_item].get('arguments'),
                            instructions_data[scenario_item].get('str_check_lambda', 'None'),
                            use_parafrases=scenarious_config.use_parafrases
                        )
                        # If you want to use the raw item structure, uncomment the next line
                        # print(f"Added scenario item: {scenarios.arguments[45]}")
        return scenarios


class AdditionalScenariusManager(ScenariousManager):

    def load(self):
        """
        Loads scenarios and encodes instructions using the provided encoding model.
        This method initializes the scenario data, encodes the instructions, and prepares the data for use in the environment.
        It retrieves the raw scenario data, transforms it into a structured format, and encodes the instructions into embeddings.
        """
        
        logger.info("Loading scenarios...")
        
        self.all_scenario = AdditioanlScenarioLoader.load_scenarios(self.config)
        
        # note: self.scenario_data.embedings_list - zero_initialized        
        self.scenario_data = ScenariousDataTransformer.transform_scenario_data(self.all_scenario)
        
        # encode the instructions use encode_model
        logger.info("Encoding instructions...")

        embeddings_list, _, _ = self.encode_instructions(self.scenario_data.instructions_list)
        self.scenario_data.embeddings_list = embeddings_list

        logger.info(f"Encoded {len(self.scenario_data.embeddings_list)} instructions.")
        
        # need JAX conversions
        self.scenario_data_jax = JaxScenariousTransformer.transform_scenario_data(self.scenario_data)
        
        self.n_instructions = 0
        logger.info(f"Final number of instructions: {len(self.scenario_data_jax.embeddings_list)}")    



def build_env():
    """Собираем среду с craftext-инструкциями и batch-обёрткой."""
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    encoder = DistilBertEncode
    scenarios = AdditionalScenariusManager
    
    env = InstructionWrapper(env, 'build_line',
                                 encode_model_class=encoder,
                                 scenario_handler_class=scenarios)
    
    return env



if __name__ == '__main__':

    a = build_env()
    print(a.scenario_handler.all_scenario.instructions_list[0])