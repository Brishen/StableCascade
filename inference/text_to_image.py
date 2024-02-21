import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.getcwd())
from utils import *
from train import WurstCoreC, WurstCoreB
from PIL import Image
from datetime import datetime
import click
import yaml


class ImageSaver:
    def __init__(self, directory: Path = Path("output"), return_images=False):
        self.directory = directory
        self.return_images = return_images
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_individual_images(self, images, **kwargs):
        now = datetime.now()

        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.size(1) > 3:
            images = images[:, :3]

        for i, img in enumerate(images):
            img = torchvision.transforms.functional.to_pil_image(img.clamp(0, 1))
            filename = f"image_{now.strftime('%Y%m%d%H%M%S')}_{i}.png"
            img.save(self.directory / filename)

    def save_image_grid(self, images, filename=f"image_grid_{datetime.now()}.png", rows=None, cols=None, **kwargs):
        now = datetime.now()
        filepath = self.directory / filename

        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.size(1) > 3:
            images = images[:, :3, :, :]

        rows = rows or 1
        cols = cols or images.size(0) // rows

        _, _, h, w = images.shape
        grid = Image.new('RGB', size=(cols * w, rows * h))

        for i, img in enumerate(images):
            img = torchvision.transforms.functional.to_pil_image(img.clamp(0, 1))
            grid.paste(img, box=(i % cols * w, i // cols * h))

        grid.save(filepath, format='PNG')

        if self.return_images:
            return grid


class ModelManager:
    def __init__(self, device):
        self.device = device
        self.core, self.core_b, self.models, self.models_b, self.extras, self.extras_b = self.load_models()

    def load_models(self):
        # Load Stage C
        core, models, extras = self.load_stage('configs/inference/trained_c_3b.yaml', WurstCoreC)

        # Load Stage B
        core_b, models_b, extras_b = self.load_stage('configs/inference/stage_b_3b.yaml', WurstCoreB)

        return core, core_b, models, models_b, extras, extras_b

    def load_stage(self, config_path, core_class):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        core = core_class(config_dict=config, device=self.device, training=False)
        extras = core.setup_extras_pre()
        models = core.setup_models(extras)
        models.generator.eval().requires_grad_(False)

        return core, models, extras


class CaptionProcessor:
    def __init__(self, image_saver: ImageSaver, model_manager):
        self.image_saver = image_saver
        self.model_manager = model_manager

    def process_caption(self, caption, output_dir, height=1024, width=1024, batch_size=4, device="cuda"):
        """
        Process a single caption to generate and save images.
        """
        # Extract the relevant instances from the model manager
        core, core_b, models, models_b, extras, extras_b = (
            self.model_manager.core,
            self.model_manager.core_b,
            self.model_manager.models,
            self.model_manager.models_b,
            self.model_manager.extras,
            self.model_manager.extras_b,
        )

        # Calculate latent sizes based on input dimensions
        stage_c_latent_shape, stage_b_latent_shape = self.calculate_latent_sizes(height, width, batch_size=batch_size)

        # Configure sampling parameters for both stages
        self.model_manager.extras.sampling_configs.update(self.configure_sampling_params(stage='C'))
        self.model_manager.extras_b.sampling_configs.update(self.configure_sampling_params(stage='B'))

        # Prepare conditions for the given caption
        conditions, unconditions = self.prepare_conditions(core, caption, batch_size, models, extras)
        conditions_b, unconditions_b = self.prepare_conditions(core_b, caption, batch_size, models_b, extras_b)

        # Perform sampling for stage C
        sampled_c = self.sample_stage_c(extras, models, conditions, unconditions, stage_c_latent_shape, device)

        # Preview image for stage c
        preview_c = models.previewer(sampled_c).float()
        self.image_saver.save_individual_images(preview_c)
        self.image_saver.save_image_grid(preview_c, "stage_c_grid.png")

        # Perform sampling for stage B using output from stage C
        sampled_b = self.sample_stage_b(extras_b, models_b, conditions_b, unconditions_b, sampled_c,
                                        stage_b_latent_shape, device)

        # Save images
        self.image_saver.save_individual_images(sampled_b)
        self.image_saver.save_image_grid(sampled_b, "stage_b_grid.png")

    def calculate_latent_sizes(self, height, width, batch_size):
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        return stage_c_latent_shape, stage_b_latent_shape

    def configure_sampling_params(self, stage):
        if stage == 'C':
            return {'cfg': 5, 'shift': 2, 'timesteps': 40, 't_start': 1.0}
        elif stage == 'B':
            return {'cfg': 1.1, 'shift': 1, 'timesteps': 40, 't_start': 1.0}
        else:
            raise ValueError("Invalid stage specified")

    def prepare_conditions(self, core, caption, batch_size, models, extras):
        batch = {'captions': [caption] * batch_size}
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False,
                                         eval_image_embeds=False)
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True,
                                           eval_image_embeds=False)
        return conditions, unconditions

    def sample_stage_c(self, extras, models, conditions, unconditions, latent_shape, device):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            sampling_c = extras.gdf.sample(
                models.generator, conditions, latent_shape,
                unconditions, device=device, **extras.sampling_configs,
            )
            for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
                sampled_c = sampled_c
        return sampled_c
        # return models.previewer(sampled_c).float()  # Assuming this gives the final image tensor

    def sample_stage_b(self, extras_b, models_b, conditions_b, unconditions_b, sampled_c, latent_shape, device):

        with (torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16)):
            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)
            sampling_b = extras_b.gdf.sample(
                models_b.generator, conditions_b, latent_shape,
                unconditions_b, device=device, **extras_b.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
                sampled_b = sampled_b

        torch.cuda.empty_cache()
        return models_b.stage_a.decode(sampled_b).float()


@click.command()
@click.argument('file', type=click.Path(exists=True), help="Caption text file for generation. One line per caption")
@click.option('--caption', type=str, default=None,
              help='Caption to process directly.')
@click.option('--output-dir', type=click.Path(), default='./output',
              help='Output directory for images.')
@click.option('--batch-size', type=int, default=4,
              help='The number of items to process in a batch.')
def main(file, caption, output_dir, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_manager = ModelManager(device=device)
    image_saver = ImageSaver(directory=Path(output_dir))
    caption_processor = CaptionProcessor(image_saver, model_manager)

    if caption:
        captions = [caption]
    elif file:
        with open(file, 'r') as f:
            captions = [caption.strip() for caption in f.readlines() if caption.strip()]
    else:
        raise click.UsageError("You must provide either a file or a caption.")

    for caption in captions:
        caption_processor.process_caption(caption, output_dir, batch_size=batch_size)


if __name__ == "__main__":
    main()
