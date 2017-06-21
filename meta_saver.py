import os
import json

def save_meta(dcgan, checkpoint_dir):
    data = {
        'output_height':dcgan.generator.output_height,
        'output_width':dcgan.generator.output_width,
        'z_dim':dcgan.generator.z_dim,
        'gf_dim':dcgan.generator.filter_dim,
        'df_dim':dcgan.discriminator.filter_dim,
        'channel':dcgan.generator.channel}
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump(data,f)

def load_meta(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'r') as f:
        return json.load(f)
