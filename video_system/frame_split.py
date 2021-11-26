import os


def load_txt(source_file):
    with open(source_file, 'r') as f:
        return f.read()


def row_qualified(row):
    width = row.replace(' ', '').split('width:')[-1].split('h')[0]
    try:
        return not row.startswith('car') and int(width) < 500
    except:
        return False


def write_txt(save_path, content):
    with open(save_path, 'w') as f:
        for row in content:
            if not row_qualified(row):
                continue
            f.write(row + '\n')


def split_txt(raw_txt, save_path):
    os.makedirs(save_path, exist_ok=True)
    file_name = '{}/front.txt'.format(save_path)
    content, frame = [], 0
    for row in raw_txt:
        if row.startswith('FPS'):
            write_txt(file_name, content)
            content, file_name = [], '{}/{}.txt'.format(save_path, frame)
            frame += 1
        content.append(row)


def main(source_file, save_path):
    raw = load_txt(source_file).split('\n')
    split_txt(raw, save_path)


if __name__ == '__main__':
    source_file = '../video_system_data/00006/00006_res.txt'
    save_path = 'frame_split'
    main(source_file, save_path)
