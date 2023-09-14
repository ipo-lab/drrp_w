from DataLoader import DataLoader

def main():
    save_path = 'results'
    loader = DataLoader(save_path)
    loader.get_ticker_list()
    loader.load_asset_data()

if __name__ == '__main__':
    main()
