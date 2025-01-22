# Image Classification using Transformers
- Dataset: "rajistics/indian_food_images"
- Check out paper 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.
- In summary, the devs adjust the input format of image dataset into vectors, then feed into transformers and transformers will handle everything, just like how transformers handle NLP downstream tasks.
- CNN process images pixel-by-pixel. For transformers, devs re-arrange them to patches, then flattened them to vectors, and that's how image dataset being feed into transformers.
