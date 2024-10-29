## Example curl command
```
curl -X POST http://127.0.0.1:5000/upload -F 'image=@/path/to/file.jpg' --output path/to/output.png
```

## Example script to collect rles

Start the server

```
python server.py ~/checkpoints/sam2 --port <your_port> --host <your_hostname> --fast
```

Collect the rles

```
xargs -I {} curl -s -w "\n" -X POST http://<your_hostname>:<your_port>/upload_rle -F 'image=@{}' < image_paths > rle_masks
```
