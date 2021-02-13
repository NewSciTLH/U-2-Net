from utils import utils
from utils import detector
from google.cloud import logging
  
from google.cloud import storage
from google.cloud import bigquery

logging_client = logging.Client()
log_name = 'Auto-colorization'
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
FROM reconciliation_input

"""
    results = query_client.query(QUERY).result()
    print(f"We have {results.total_rows} folders of orders in the directories socks, koozies and stickers")
    key, key_m, source_blob_name, source_blob_name_m, bucket, 'human'
    to_list = [ {"key":item['key'],
                 "key_m":item['key_m'],
                 "bucket": item['simple_crop'].split('/')[0],
                 "mask":item['simple_crop'].replace(f'{item['simple_crop'].split('/')[0]}/',''),
                 "crop":item['final_crop'].replace(f'{item['simple_crop'].split('/')[0]}/',''),
                 "classes":item['subject_class']}]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        executor.map(utils.human_eyes, to_list)
    print('in dataDec/ you can find the files organized by day')

if __name__ == "__main__":
    