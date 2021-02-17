from utils import utils
from utils import detector
from google.cloud import logging
import os  
from google.cloud import storage
from google.cloud import bigquery
import threading
import concurrent.futures

logging_client = logging.Client()
log_name = 'Reconciliation'
logger = logging_client.logger(log_name)

if os.path.isfile('/home/ericd/bqkey.json'): 
    query_client = bigquery.Client.from_service_account_json("/home/ericd/bqkey.json")
    storage_client = storage.Client.from_service_account_json("/home/ericd/storagekey.json")
else:
    query_client = bigquery.Client()
    storage_client = storage.Client()
    
print('Downloading weights...')

if not os.path.isfile('checkpoints/cat_model_epoch_015.pth'):
    utils.downloadBlob('model_staging','eyes/ResNet50_Models/cat_model_epoch_015.pth', 'checkpoints/cat_model_epoch_015.pth')
    utils.downloadBlob('model_staging','eyes/ResNet50_Models/dog_model_epoch_015.pth', 'checkpoints/dog_model_epoch_015.pth')
    utils.downloadBlob('model_staging','eyes/ResNet50_Models/human_model_epoch_015.pth', 'checkpoints/human_model_epoch_015.pth')
print('done')


def create_data():
    """Query those images in the folders socks, koozies and stickers by time of creation"""
    QUERY="""SELECT * 
FROM newsci-1532356874110.divvyup_metadata.reconciliation_input
"""
    results = query_client.query(QUERY).result()
    print(f"We have {results.total_rows} folders of orders in the directories socks, koozies and stickers")
    #key, key_m, source_blob_name, source_blob_name_m, bucket, 'human'
    to_list = [ {"key":str(item['key']),
                 "key_m":str(item['mask_key'])+'m',
                 "bucket": 'divvyup_data',
                 "crop":item['simple_crop'],
                 "mask":item['final_crop'],
                 "classes":item['subject_class']} for item in results if item['subject_class']
              ]
    print(f'We have {len(to_list)} valid inputs')
    for d_c in to_list:
        utils.human_eyes(d_c)
    #with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    #    executor.map(utils.human_eyes, to_list)
    print('program ended')

if __name__ == "__main__":
    print('test')
    create_data()
    print('test ended')
    