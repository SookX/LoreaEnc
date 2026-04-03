set -e
mkdir -p datasets/librispeech
cd datasets/librispeech
echo "Downloading LibriSpeech dataset..."
wget -c https://www.openslr.org/resources/12/test-clean.tar.gz

