"""Prepare Datasets for Translation Tasks"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/translation')
_DOWNLOAD_ZIP_FILES = dict(
    ae_photos='0888f47b7ae782a28046025ef8a037162a214648',
    apple2orange='bd1a3b479e950d65089100cf2e86d9cf06924714',
    cezanne2photo='2d72f5b4c228ade7f272a7aa337c1e4debd40d41',
    cityscapes='c44745238cee71d8696a17326bafa42ce74e9103',
    facades='a841c173ae9e303dc8a07877c68f12bc28d79e71',
    horse2zebra='f4d168a5f286eea57bce042ecbd8f8a3b95c2b21',
    iphone2dslr_flower='ac67208b72e915424dab0b455a622fbe92613441',
    maps='d6feccffec64f8fc222e9afeef6664802bb50d90',
    monet2photo='7fd5f0fde0f63087809b0ed943a65ffb2fcacc3c',
    summer2winter_yosemite='697b6ea004d7379575f82ba4912673eff1cb3bf9',
    ukiyoe2photo='13641096debf3c121474596a094d458c2f7832a0',
    vangogh2photo='e55e6cd419b8468c88f724101d740a25a09f8728')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Translation datasets.',
        epilog=' '.join(['Example:', 'python', 'translation.py',
                         '--download-dir', '~/TranslationDevkit',
                         '--all']),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--download-dir', type=str,
                        default='~/TranslationDevkit/',
                        help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true',
                        help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help=' '.join(['overwrite downloaded files if set,',
                                       'in case they are corrputed']))
    parser.add_argument('--all', action='store_true',
                        help='download all available datasets')
    parser.add_argument('--dataset', action='store', nargs='+',
                        help=' '.join(
                            ['specify datasets to be downloaded.',
                             'Available options:'] +
                            ['{},'.format(filename) for filename
                                in _DOWNLOAD_ZIP_FILES.keys()]))
    args = parser.parse_args()
    return args

################################################################################
# Download and extract traslation datasets into ``path``


def download_translation(path, datasets, overwrite=False):
    _DOWNLOAD_URL = ''.join(['https://people.eecs.berkeley.edu/',
                             '~taesung_park/CycleGAN/datasets/'])
    _DOWNLOAD_URLS = [(''.join([_DOWNLOAD_URL, dataset, '.zip']),
                       _DOWNLOAD_ZIP_FILES[dataset]) for dataset in datasets]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite,
                            sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename) as _zip:
            _zip.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.download_dir)
    if args.dataset is None or args.all:
        datasets = list(_DOWNLOAD_ZIP_FILES.keys())
    else:
        datasets = [dataset for dataset in args.dataset
                    if dataset in _DOWNLOAD_ZIP_FILES]

    print("Datasets to be processed: {}".format(', '.join(datasets)))

    if not os.path.isdir(path) or args.overwrite:
        if args.no_download:
            raise ValueError(('{} is not a valid directory,'
                              ' make sure it is present.'
                              ' Or you should not specify'
                              ' "--no-download" to grab it'.format(path)))
        else:
            download_translation(os.path.join(path, 'TranslationDevkit'),
                                 datasets, overwrite=args.overwrite)
            for dataset in datasets:
                shutil.move(os.path.join(path, 'TranslationDevkit', dataset),
                            os.path.join(path, dataset))
            shutil.rmtree(os.path.join(path, 'TranslationDevkit'))

    # make symlink
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(path, _TARGET_DIR)
