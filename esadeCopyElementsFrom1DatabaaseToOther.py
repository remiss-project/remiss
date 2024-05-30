﻿# code to get info from MongoDB database
from pymongo.mongo_client import MongoClient

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from tqdm import tqdm


def connect_to_mongo(uri, db_name, collection_name):
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        return collection
    except PyMongoError as e:
        print(f"Error connecting to MongoDB: {e}")
        return None


def transfer_documents(source_uri, source_db, source_col, dest_uri, dest_db, dest_col):
    source_collection = connect_to_mongo(source_uri, source_db, source_col)
    if source_collection is None:
        return

    try:
        result_source = source_collection.find()
    except PyMongoError as e:
        print(f"Error finding documents: {e}")
        return

    print("Elements Loaded")
    dest_collection = connect_to_mongo(dest_uri, dest_db, dest_col)
    if dest_collection is None:
        return
    total = 0
    for i, doc in tqdm(enumerate(result_source)):
        try:
            dest_collection.insert_one(doc)
            total += 1

        except PyMongoError as e:
            print(f"Error inserting document: {e}")

    print(f'Total elements transferred: {total}')


def main():
    uriSource = "mongodb+srv://esala:eureka@cvcui.n1hmtmt.mongodb.net/?retryWrites=true&w=majority"
    # uriDestiny = "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.0.0"
    uriDestiny = "mongodb://srvinv02.esade.es:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.0.0"

    source_collection = "CVCFeatures"
    target_collection = "profiling"

    database_mapping = {
        "Barcelona_2019": "CVCUI_BCN19",
        "Generales_2019": "CVCUI_ESP19",
        "Generalitat_2021": "CVCUI_GEN21",
        "MENA_Agressions": "CVCUI_VIOLA",
        "MENA_Ajudes": "CVCUI_AJUDES",
        "Openarms": "CVCUI_OPENMAFIA"
    }

    for target_database, source_database in database_mapping.items():
        print(f"Transferring from {source_database} to {target_database}")
        transfer_documents(uriSource, source_database, source_collection, uriDestiny, target_database,
                           target_collection)


if __name__ == "__main__":
    main()
