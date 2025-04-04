from options import parser
import pytorch_lightning as pl
from dataset import VQADataModule


def main():
    args = parser.parse_args()
    data_module = VQADataModule(args)


if __name__ == "__main__":
    main()
