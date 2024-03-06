# code to get info from MongoDB database
from pymongo.mongo_client import MongoClient

uriSource = "mongodb+srv://esala:eureka@cvcui.n1hmtmt.mongodb.net/?retryWrites=true&w=majority"
uriDestiny = "mongodb+srv://esala:eureka@cvcui.n1hmtmt.mongodb.net/?retryWrites=true&w=majority"
try:
    myclientSource = MongoClient(uriSource)
    mydbSource = myclientSource['CVCUI']
    mycolSource = mydbSource["CVCFeatures"]

    # result = mycol.find().limit(5)
    resultSource = mycolSource.find()
except Exception as e:
    print(e)

print("Elements Loaded")

if resultSource:
    for i, doc in enumerate(resultSource):
        all_features_for_user = doc
        print("Document found with index", i)
        print(all_features_for_user["twitter_id"])

        try:
            myclientDestiny = MongoClient(uriDestiny)
            mydbDestiny = myclientDestiny['CVCUI2']
            mycolDestiny = mydbDestiny["CVCFeatures2"]
            mycolDestiny.insert_one(doc)
            print("INSERTED")
        # print (x)
        except Exception as e:
            print(e)
    print("Bye")
