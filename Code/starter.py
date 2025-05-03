from Training import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains an the piano model. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=[" "], type=str, nargs='+', help='The names of the datasets to use. Datasets = [CL1BTapePreamp, TapePreamp, CL1BTape, CL1BPreamp].')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--steps', default=240, type=int, nargs='?', help='Number of steps to generate.')

    parser.add_argument('--harmonics', default=24, type=int, nargs='?', help='Number of harmonics to synthetize.')

    parser.add_argument('--scenario', default='1', type=str, nargs='+', help='Scenario to evaluate: unseen key (1),  unseen velocity (2)')

    parser.add_argument('--learning_rate', default=1e-6, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):


    print("######### Preparing for training/inference #########")
    print("\n")

    if args.only_inference:
        train(data_dir=args.data_dir,
              model_save_dir=args.model_save_dir,
              save_folder=f'{args.dataset}_{args.harmonics}',
              learning_rate=args.learning_rate,
              epochs=args.epochs,
              steps=args.steps,
              harmonics=args.harmonics,
              phase='A',
              scenario=args.scenario,
              inference=True)
    else:
        train(data_dir=args.data_dir,
                model_save_dir=args.model_save_dir,
                save_folder=f'{args.dataset}_{args.harmonics}',
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                steps=args.steps,
                harmonics=args.harmonics,
                phase='B',
                scenario=args.scenario,
                inference=args.only_inference)

        train(data_dir=args.data_dir,
              model_save_dir=args.model_save_dir,
              save_folder=f'{args.dataset}_{args.harmonics}',
              learning_rate=args.learning_rate,
              epochs=args.epochs,
              steps=args.steps,
              harmonics=args.harmonics,
              phase='A',
              scenario=args.scenario,
              inference=args.only_inference)

def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()