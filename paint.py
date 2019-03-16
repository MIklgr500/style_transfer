import os
import warnings
import argparse
from keras.preprocessing.image import save_img
import matplotlib.pyplot as plt

from config import ArtistConfig
from artist import Artist
from models import ArtistVGG19


def main(args):
    model = ArtistVGG19(shape=(args.height, args.width, 3),
                        verbose=args.verbose)
    config = ArtistConfig(contest_path=os.path.join(args.content_path, args.content_img),
                          content_layers=[2, 7, 12, 21, 30],
                          content_layer_weights=[1/5., 1/5., 1/5., 1/5., 1/5.],
                          style_path=os.path.join(args.style_path, args.style_img),
                          style_layers=[23],
                          style_layer_weights=[1],
                          size=(args.height, args.width),
                          alpha=args.alpha,
                          beta=args.beta,
                          gamma=args.gamma,
                          lr=args.lr,
                          n_iter=args.n_iter,
                          noise_rate=args.noise_rate,
                          optimizer='adam',
                          debug=args.debug,
                          verbose=args.verbose)

    artist = Artist(model=model)
    img = artist.transform(config=config)

    save_img(os.path.join('result', args.img_name + '.jpg'), img)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    desc = "TensorFlow implementation of 'A Neural Algorithm for Artistic Style'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--img_name', type=str, default='result')
    parser.add_argument('--style_img', default='style4.jpg', type=str)
    parser.add_argument('--content_img', default='content.jpg', type=str)
    parser.add_argument('--style_path', default='data/style', type=str)
    parser.add_argument('--content_path', default='data/content', type=str)
    parser.add_argument('--lr', default=5., type=float)
    parser.add_argument('--alpha', default=5*1e-3, type=float)
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--gamma', default=1e-2, type=float)
    parser.add_argument('--noise_rate', default=0.75, type=float)
    parser.add_argument('--height', default=324, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--n_iter', default=500, type=int)
    parser.add_argument('--debug', default=False, type=bool)

    args = parser.parse_args()
    main(args)
