import requests
from ckanapi import RemoteCKAN
import json
import os
import xlrd
import csv


def excel_to_csv(excel_filename, csv_filename):
    with xlrd.open_workbook(excel_filename) as workbook:
        assert len(workbook.sheets) == 1
        sh = workbook.sheet_by_index(0)
        with open(csv_filename,'w') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            for row in range(sh.nrows):
                writer.writerow(sh.row_values(row))


def strip_empty(to_strip):
    if isinstance(to_strip, str) or isinstance(to_strip, bool) or isinstance(to_strip, int):
        return to_strip

    elif isinstance(to_strip, list):
        new_list = [strip_empty(val) for val in to_strip if val]
        if to_strip == new_list:
            return new_list
        else:
            return strip_empty(new_list)

    elif isinstance(to_strip, dict):
        new_dict = {key: strip_empty(val) for (key, val) in to_strip.items() if val}
        if to_strip == new_dict:
            return new_dict
        else:
            return strip_empty(new_dict)

    else:
        raise Exception('invalid input type: {0}, should be string or dict'.format(type(to_strip)))

def get_dataset_name(dataset, safechars=['-', '.', '_']):
    name = dataset['title'].lower().replace(' ', '-')
    name = ''.join(c for c in name if c.isalnum() or c in safechars)
    return name.replace('----', '-').replace('---', '-').replace('--', '-')

def save_metadata(dataset, dataset_folder):  #  dataset_name=None, data_dir='data/ckan'):
    dataset_name = dataset_folder.split('/')[-1]  # get_dataset_name(dataset)
    filename = '{0}/{1}_metadata.json'.format(dataset_folder, dataset_name)
    with open(filename, 'w') as json_file:
        metadata = strip_empty(dataset)
        json.dump(metadata, json_file, indent=2)


def is_valid_resource(resource, formats=['xls', 'xlsx', 'csv']):
    return resource['format'].lower() in formats # and (resource['size'] > 0)


def scrape_ckan_instance(ckan_url='https://open.alberta.ca', formats=['xls', 'xlsx', 'csv'], data_dir='data/ckan'):

    print('scraping ckan instance', ckan_url)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    instance = RemoteCKAN(ckan_url)
    print('retrieving list of instance datasets')
    datasets = instance.action.current_package_list_with_resources()  # limit, offset)

    print('processing datasets')
    for dataset in datasets:
        # print(dataset['title'])
        # check that at least one resource is in a desired format
        valid_resources = [resource for resource in dataset['resources'] if is_valid_resource(resource, formats)]
        if valid_resources:
            dataset_name = get_dataset_name(dataset)
            dataset_folder = '{0}/{1}'.format(data_dir, dataset_name)
            if not os.path.isdir(dataset_folder):
                os.mkdir(dataset_folder)

            save_metadata(dataset, dataset_folder)

            for resource in valid_resources:
                process_resource(resource, dataset_folder, formats=formats)


def process_resource(resource, dataset_folder, formats=['xls', 'xlsx', 'csv']):
    try:
        # if filetype not in formats, return 
        if not resource['url'].split('.')[-1].lower() in formats:
            print('invalid filetype, resource url:', resource['url'])
            return
        
        resource_fname = resource['url'].split('/')[-1]
        data_filename = '{0}/{1}'.format(dataset_folder, resource_fname)

        # if file already exists and isnt empty, return
        if os.path.isfile(data_filename) and os.path.getsize(data_filename) > 0:
            print('resource already present:', data_filename)
            return
        
        # o.w. request resource from url
        response = requests.get(resource['url'], stream=True)

        if response.status_code == 200:
            # if successful, stream data to file
            print('saving file:', data_filename)
            with open(data_filename, 'wb') as data_file:
                for chunk in response.iter_content(): 
                    # if chunk:
                    data_file.write(chunk)
            
        else:
            print('request failed:', response.status_code)
            print('failed resource:', strip_empty(resource))
    
    except requests.exceptions.InvalidSchema as err:
        print('invalid shema, not http? url:', resource['url'])
        print('error:', err)
    
    except Exception as err:
        print('unknown error:', err)


def main(data_dir='data/ckan', formats=['xls', 'xlsx', 'csv']):
    # read in list of ckan instances
    with open('{0}/ckan-instances.json'.format(data_dir)) as json_file:
        instances = json.load(json_file)
    
    for ckan_url in instances:
        try:
            scrape_ckan_instance(ckan_url=ckan_url, formats=formats, data_dir=data_dir)

        except Exception as err:
            print('error scraping ckan instance:', ckan_url)
            print('error:', err)
    

if __name__ == '__main__':
    # main()
    scrape_ckan_instance(ckan_url="https://catalogue.data.gov.bc.ca")
