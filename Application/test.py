import os
import json
from database import *


db = DataBase(path=os.path.join(os.getcwd(), "database"), filename="DigiDoc.db")
db.create_table(name="users",
                labels={"id": DBType.TEXT,
                        "email": DBType.TEXT,
                        "password": DBType.TEXT,
                        "files": DBType.TEXT},
                primary_key="id")


files = json.loads(db.get_table("users").get_from_cell(key=str(0), column_name="files"))
files[10] = {"type": "pdf", "name": "some.pdf", "data": "somedata"}
db.get_table("users").set_to_cell(key=str(0), column_name="files", new_value=json.dumps(files))

