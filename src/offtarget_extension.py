import copy
import logging
import os
from glob import glob
import tensorflow as tf
import knime_extension as knext
import numpy as np
from py4j.java_gateway import JavaGateway
import pandas as pd

LOGGER = logging.getLogger(__name__)

columns_names = ['smiles', 'molecules', 'structures', 'mols', 'smi', 'canonical_smiles', 'canonisized_smiles',
                 'canon_smiles',
                 'can_smiles', 'can_smi']


root = os.path.dirname(os.path.abspath(__file__))
list_of_models = {path.split('/')[-1].split('.')[0]:path for path in glob(os.path.join(root,'models',"*.h5"))}

pharminfo = knext.category(
    path="/community",
    level_id="pharmacoinformatics",
    name="Pharmacoinformatics Research Group (UNIVIE)",
    description="Nodes created by the Pharmacoinformatics Research Group at the University of Vienna members.",
    icon="univie.png",
)

class ECFP4Calc():
    def __init__(self):

        self.gateway = JavaGateway.launch_gateway(
            classpath=os.path.join(root,'cdk-2.7.1.jar'),
            die_on_exit=True)
        cdk = self.gateway.jvm.org.openscience.cdk
        builder = cdk.DefaultChemObjectBuilder.getInstance()
        self.circularFingerprints = cdk.fingerprint.CircularFingerprinter(3)# 3 is for ECFP4 (see https://github.com/CDK-R/cdkr/blob/cef1eed1555947ed82e8303cb6d79c77fd89b3c1/rcdk/R/fingerprint.R#L81)
        self.smiles_parser = cdk.smiles.SmilesParser(builder)
    def get_bit_vector(self,indexes):
        idx = np.zeros(1024) #Just pre-defined as in Naga's code
        idx[indexes] = 1
        return idx

    def __call__(self, smiles):
        mol = self.smiles_parser.parseSmiles(smiles)
        ecfp4 = (np.array((self.circularFingerprints.getBitFingerprint(mol).getSetbits())))
        return self.get_bit_vector(ecfp4)

calculator = ECFP4Calc()

@knext.node(name="Off-target", node_type=knext.NodeType.MANIPULATOR, icon_path="offtarget.ico", category=pharminfo)
@knext.input_table(name="Molecular Table", description="A dataset with molecular structures")
@knext.output_table(name="Processed Molecules", description="The same datatable, plus all columns with predictions")

class OffTargetKnimeNode():
    smiles_column = knext.ColumnParameter(label="SMILES column",description="The SMILES column to use", column_filter=lambda x: (x.name.lower() in columns_names), port_index=0,include_none_column=False )
    def configure(self, configure_context, input_schema):
        output_schema = input_schema
        for endpoint in list_of_models.keys():
            output_schema = output_schema.append(knext.Column(knext.double(), endpoint))
        return output_schema

    def execute(self, exec_context, input_1):
        def robust(x):
            try:
                return calculator(x)
            except:
                return np.nan
        data = input_1.to_pandas()
        if self.smiles_column == None:
            # Info
            LOGGER.info("No smiles column found. Trying to guess one...")
            smiles_column = [x for x in data.columns if x.lower() in columns_names]
            assert len(smiles_column) == 1, "Dataframe should contain ONLY one smiles column, but found: {}. Please, " \
                                            "choose the column in configuration menu.".format(
                smiles_column)
            smiles_column = smiles_column[0]
        else:
            smiles_column = self.smiles_column

        data['ecfp4'] = data[smiles_column].apply(robust)
        correct = data[~data['ecfp4'].isna()]
        LOGGER.info("Is na {}".format(data['ecfp4'].isna()))
        for model_name, model_path in list_of_models.items():
            with tf.device('/cpu:0'):
                model = tf.keras.models.load_model(model_path, custom_objects={'balanced_acc': None})
                res = model.predict(np.stack(correct['ecfp4'].values)).flatten().astype(np.float64)
                predicted = pd.DataFrame(res, index=correct.index, columns=[model_name])
                data = pd.concat([data,predicted],axis=1) #pd.concat([correct, pd.Series(res, name=model_name)], axis=1)
        data = data.drop(columns=['ecfp4'])
        return knext.Table.from_pandas(data)


