from datadotworld.client import api as ddw
import requests
import json
import pandas as pd
import boto3


TOKEN='eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwcm9kLXVzZXItY2xpZW50OmNyYWlnLWNvcmNvcmFuIiwiaXNzIjoiYWdlbnQ6Y3JhaWctY29yY29yYW46OjE1Mjg2MmI3LTkxMmEtNDk5MC1iMWUwLWExZmExMDRiNDU0NCIsImlhdCI6MTUxNzIzNDY3MSwicm9sZSI6WyJ1c2VyX2FwaV9yZWFkIiwidXNlcl9hcGlfd3JpdGUiXSwiZ2VuZXJhbC1wdXJwb3NlIjp0cnVlfQ.1pil6Y7L27-MLAgjKm9mD-ebOyDLRygJ7w88pgdCXbccatZNzhHDhWVd1QygER9Vj7Szpw6hOn56ocnpqwM8Vg'


def main(user='craig-corcoran', project='dataset-labeling'):

    req_params = {'headers': {'Authorization':'Bearer {0}'.format(TOKEN)}}
    response = requests.get('https://api.data.world/v0/projects/{0}/{1}'.format(user, project), **req_params)
    content = json.loads(response.content)

    datasets = content['linkedDatasets']

    base_url = 'https://api.data.world/v0'

    tags = {}
    for ds in datasets:
        key = '{0}/{1}'.format(ds['owner'], ds['id'])
        file_key = key.replace('/', '_')
        print('processing dataset:', file_key)

        dat_url = '{0}/datasets/{1}'.format(base_url, key)
        response = requests.get(dat_url, **req_params)
        ds_content = json.loads(response.content)
        tags[key] = ds_content['tags']

        with open('data/ddw/{0}_tags.json'.format(file_key), 'w') as json_file:
            json.dump(ds_content['tags'], json_file)

        # get table names
        table_query = 'SELECT * FROM Tables'
        sql_url = '{0}/sql/{1}'.format(base_url, key)
        tables = requests.get(sql_url, params={'query': table_query}, **req_params)
        tables = json.loads(tables.content)

        for table in tables:
            if table: 
                table_name = table['tableId']
                if table_name:
                    # print('reading from table:', table_name)
                    data_query = 'SELECT * FROM {0}'.format(table_name)
                    data = requests.get(sql_url, params={'query': data_query}, **req_params)  # , 'includeTableSchema': False (in params)
                    if data.status_code == 200:
                        try:
                            data = json.loads(data.content)
                            df = pd.DataFrame(data)
                            data_filepath = 'data/ddw/{0}_{1}.csv'.format(file_key, table_name)
                            # print('saving to file:', data_filepath)
                            df.to_csv(data_filepath, index=False)
                        
                        except Exception as e:
                            print('error:', e)
                    else:
                        print('request failed:', data.__dict__)
                else:
                    print('missing table id:', table)
            else:
                print('missing table in:', tables)

def read_s3():

                        
        
if __name__ == '__main__':
    main()