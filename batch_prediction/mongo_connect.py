import pymongo
import pandas as pd

class mongo_connect:
    def mongo_connect(mongo_connection, mongo_database, mongo_collection):
        #print(mongo_connection)
        client = pymongo.MongoClient(str(mongo_connection), tls=True, tlsAllowInvalidCertificates=True)

        db = client[str(mongo_database)]
        collection = db[str(mongo_collection)]
        l = []
        for i in collection.find():
            l.append(i)
            # print(i)
        df = pd.DataFrame(l)
        return df