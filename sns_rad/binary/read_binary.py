from . import ReadStorage
import argparse
from argparse import RawTextHelpFormatter


def main():
    epilog_text = '''
        Examples: 
                python read_file.py filename   
                # will read and print out all records from <filename>           
    
                python  read_file.py -i n1 -i n2 filename 
                # will read and print out records with indices n1 and n2 from <filename> 
    '''
    parser = argparse.ArgumentParser(epilog=epilog_text, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--index', '-i', action='append', default=[], type=int,
                        help='indices of records to read')
    parser.add_argument('file', help='file to read')
    args = parser.parse_args()

    file = ReadStorage(args.file)
    print(args.file, 'has', len(file), 'records')


    def show(record):
        if 'value' in record:
            value = record['value']
            # if isinstance(value,(str,list,tuple)):
            if hasattr(value, '__len__') and len(value) > 4:
                record['value'] = [value[0], value[1], '...', value[len(value) - 1]]
        return record

    if not args.index:
        for record in file:
            print(show(record))
    else:
        for i in args.index:
            print(i, ":", show(file[i]))

    file.close()

if __name__ == "__main__":
    main()
