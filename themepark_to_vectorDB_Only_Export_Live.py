import requests
import pandas as pd
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.document_loaders import DataFrameLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from datetime import datetime
import pytz 
import os

# Load OpenAI API key and Qdrant settings
api_key = os.getenv("MY_OPENAI_KEY") or "default-fake-key-for-testing"
if api_key == "default-fake-key-for-testing":
    print("Warning: Using default key. Check environment variables.")
else:
    # Set to the OpenAI environment variable expected by the library
    os.environ["OPENAI_API_KEY"] = api_key

qdrant_url = os.getenv("MY_QDRANT_URL")
qdrant_key = os.getenv("MY_QDRANT_KEY")

# Ensure Qdrant variables are set
if not qdrant_url or not qdrant_key:
    raise ValueError("Qdrant environment variables are not set. Please set them in GitHub Secrets or locally.")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize the Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_key
)

# Define the base URL for the API
base_url = "https://api.themeparks.wiki/v1"

# List of entity IDs for parks
entity_ids = ['73436fe5-1f14-400f-bfbf-ab6766269e70', '9a268a42-4f13-4f7a-9c7d-d40a20d7a6eb', 'a148a943-616b-41c8-b5f8-27e67f7bdf33', '7fa8dc48-1240-4c57-b576-e4eca4bf4343', '18635b3e-fa23-4284-89dd-9fcd0aaa9c9c', 'f0ea9b9c-1ccb-4860-bfe6-b5aea7e4db2b', 'bdf9b533-144c-4b78-aa2f-5173c5ce5e85', '58392c29-d79d-49e4-9c35-0100d417d24e', '639738d3-9574-4f60-ab5b-4c392901320b', 'd21fac4f-1099-4461-849c-0f8e0d6e85a6', '28aee1df-1d05-4f53-bbf5-08f7aabff3a1', '7502308a-de08-41a3-b997-961f8275ab3c', 'abb67808-61e3-49ef-996c-1b97ed64fac6', '3ad67aac-0b97-4961-8aa8-c3b8d2350873', '18a2aeb2-7be5-4273-8cac-611ae5b519f6', '6f8764b7-172a-4fcf-8fec-10d3e44a55e4', '5f28a518-dfb4-4650-a642-2d88c8e30d8e', '99e7087b-951a-4322-84a8-572f26bc324d', '75ea578a-adc8-4116-a54d-dccb60765ef9', '47f90d2c-e191-4239-a466-5892ef59a88b', '288747d1-8b4f-4a64-867e-ea7c9b27bad8', '1c84a229-8862-4648-9c71-378ddd2c7693', 'b070cbc5-feaa-4b87-a8c1-f94cca037a18', 'ead53ea5-22e5-4095-9a83-8c29300d7c63', 'de3955e7-e3b7-4b85-884b-2d0fb2c1d71c', 'b4dd937f-a79d-4b82-922f-e8ab0fbf5b5b', 'fe78a026-b91b-470c-b906-9d2266b692da', '267615cc-8943-4c2a-ae2c-5da728ca591f', 'eb3f4560-2383-4a36-9152-6b3e5ed6bc57', 'e35cd439-ca3c-4234-ad92-d699995d0cba', '0f044655-cd94-4bb8-a8e3-c789f4eca787', 'bc4005c5-8c7e-41d7-b349-cdddf1796427', '4535960b-45fb-49fb-a38a-59cf602a0a9c', '66e12a41-3a09-40cd-8f55-8d335d9d7d93', '93142d7e-024a-4877-9c72-f8e904a37c0c', '000c724a-cd0f-41a1-b355-f764902c2b55', '67b290d5-3478-4f23-b601-2f8fb71ba803', '3cc919f1-d16d-43e0-8c3f-1dd269bd1a42', '30713cf6-69a9-47c9-a505-52bb965f01be', 'bb731eae-7bd3-4713-bd7b-89d79b031743', 'c8299e1a-0098-4677-8ead-dd0da204f8dc', 'f4bd1a23-44f0-444b-a91c-8d24f6ec5b1f', '9e938687-fd99-46f3-986a-1878210378f8', '15805a4d-4023-4702-b9f2-3d3cab2e0c1e', 'd4c88416-3361-494d-8905-23a83e9cb091', '589627eb-fe16-4373-a2db-08d73805fb1f', '0c7ab128-259a-4390-93b9-d2e0233dfc16', '95162318-b955-4b7e-b601-a99033aa0279', '75122979-ddea-414d-b633-6b09042a227c', 'dd0e159a-4e4b-48e5-8949-353794ef2ecb', '27d64dee-d85e-48dc-ad6d-8077445cd946', '9e2867f8-68eb-454f-b367-0ed0fd72d72a', '19d7f29b-e2e7-4c95-bd12-2d4e37d14ccf', '1989dca9-c8d3-43b8-b0dd-e5575f692b95', 'ca888437-ebb4-4d50-aed2-d227f7096968', 'dae968d5-630d-4719-8b06-3d107e944401', 'f9c2e042-8604-4fe8-9909-e8f95f0942f5', '32608bdc-b3fa-478e-a8c0-9dde197a4212', 'd06d91b8-7702-42c3-a8af-7d0161d471bf', '91c92c4c-e079-4488-8c99-385bc81bd5d7', '815e6367-9bbe-449e-a639-a093e216188f', 'd553882d-5316-4fca-9530-cc898258aec0', 'd67e40f9-9c02-4bfe-8ee1-b714deda9906', 'd2bef7bc-f9fc-4272-a6f1-2539d7413911', '556f0126-8082-4b66-aeee-1e3593fed188', 'c6073ab0-83aa-4e25-8d60-12c8f25684bc', '8be1e984-1e5f-40d0-a750-ce8e4dc2e87c', '3237a0c2-8e35-4a1c-9356-a319d5988e7c', 'ab49b801-9b07-4cbc-9b3e-9896e538872e', '98f634cd-c388-439c-b309-960f9475b84d', 'fc40c99a-be0a-42f4-a483-1e939db275c2', 'e9805d65-edad-4700-8942-946e6a2b4784', '164f3ee7-5fd7-47ac-addc-40b5e3e2b144', '722116aa-56be-4466-8c6f-a5acbac05da2', '66f5d97a-a530-40bf-a712-a6317c96b06d', 'a0df8d87-7f72-4545-a58d-eb8aa76f914b', '0a6123bb-1e8c-4b18-a2d3-2696cf2451f5', '24cdcaa8-0500-4340-9725-992865eb18d6', 'ddc4357c-c148-4b36-9888-07894fe75e83', '68e1d8f0-ed42-4351-af25-160421e37ce0', '7340550b-c14d-4def-80bb-acdb51d49a66', '832fcd51-ea19-4e77-85c7-75d5843b127c', 'bd0eb47b-2f02-4d4d-90fa-cb3a68988e3b', '043211c0-76f2-4456-89f8-4001be01018d', 'bb285952-7e52-4a07-a312-d0a1ed91a9ac', 'b08d9272-d070-4580-9fcd-375270b191a7', 'a4f71074-e616-4de4-9278-72fdecbdc995', '0d8ea921-37b1-4a9a-b8ef-5b45afea847b', 'ae959d1f-9fcc-4aab-8063-71e641fa57f4', '6535c36f-ea51-4156-824e-f304b27fb1f6']


# Function to fetch live data for a specific entity
def get_live_data(entity_id):
    # Construct the URL for the live data endpoint
    url = f"{base_url}/entity/{entity_id}/live"
    attractions_with_wait_time = []  # List to hold attractions data

    try:
        # Send a GET request to the API
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            pacific_time = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')  #!!!!!

            # Extract live data
            live_data = data.get('liveData', [])
            # Filter attractions with available wait times
            for attraction in live_data:
                if 'queue' in attraction and 'STANDBY' in attraction['queue'] and 'waitTime' in attraction['queue']['STANDBY']:
                    # Collect relevant data
                    attraction_info = {
                        'id': attraction.get('id'),
                        'name': attraction.get('name'),
                        'wait_time': attraction['queue']['STANDBY']['waitTime'],
                        'entity_type': attraction.get('entityType'),
                        'status': attraction.get('status'),
                        'last_updated': pacific_time  # Updated to use current Pacific Tim
                    }
                    attractions_with_wait_time.append(attraction_info)
        else:
            print(f"Error fetching data for ID {entity_id}: {response.status_code}")
    except Exception as e:
        print(f"An error occurred for ID {entity_id}: {e}")

    return attractions_with_wait_time

def fetching_live_data(all_attractions):
    #counter = 0  # Initialize a counter
    for entity_id in entity_ids:
        print(f"\nFetching live data for entity ID: {entity_id}")
        attractions = get_live_data(entity_id)
        all_attractions.extend(attractions)
    return(all_attractions)

def sent_to_vector(all_attractions):
    # Convert each attraction dictionary into a Document object
    documents = [
        Document(
            page_content=attraction["name"],
            metadata={
                "id": attraction["id"],
                "wait_time": attraction["wait_time"],
                "entity_type": attraction["entity_type"],
                "status": attraction["status"],
                "last_updated": attraction["last_updated"]
            }
        )
        for attraction in all_attractions
    ]
    
    # Get a list of all collections
    collections = qdrant_client.get_collections()
    for collection in collections.collections:
        collection_name = collection.name
        print(f"Dropping collection: {collection_name}")
        qdrant_client.delete_collection(collection_name)
    # Send to the vector db
    qdrant = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="chatbot",
        api_key=qdrant_key
    )
    print(f"Sucess sent to vector")
    
# Main function to fetch data and export it
def main():
    all_attractions = []  # List to hold all attractions from all parks
    fetching_live_data(all_attractions)
    # Export collected data to Excel
    if all_attractions:
        #export_to_excel(all_attractions)
        sent_to_vector(all_attractions)
    else:
        print("No attractions with wait times found.")
        
# Execute the main function
if __name__ == "__main__":
    main()
    


# In[ ]:




