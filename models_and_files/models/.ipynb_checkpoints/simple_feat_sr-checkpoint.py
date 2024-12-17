class InterpolationModel(nn.Module):
    def __init__(self):
        super(InterpolationModel, self).__init__()
        #ResNet18 for feature extraction
        self.encoder = resnet18(weights=None)  # No pretrained waits for this trial run
        self.encoder.conv1 = nn.Conv2d(502, 64, kernel_size=5, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  #fc layer will flatten - we don't need that for feature extraction spatial

        #Upsample to match [313, 313] which you need for the REsUnet pipeline
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1), #[25 x 25] to [50 x 50]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),# [32x32] to [64x64]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),#[64x64] to [128x128]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=1),#[128x128] to [256x256]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 502, kernel_size=2, stride=1, padding=1), #[256x256] to [256x256]
            nn.Upsample(size=(313, 313), mode='bilinear', align_corners=True) #Final upscale to [313x313]
        )

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        features = features.view(features.size(0), -1, 1, 1)  # Reshape for decoding
        output = self.decoder(features)  # Decode to high-resolution size
        return output