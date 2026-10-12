# Dictation into a textarea

The app the library's README describes, complete. It is not part of the package; copy it somewhere and:

```sh
bun add /path/to/parakeet-web-0.1.0.tgz
bunx parakeet-web-assets public/parakeet-web
mkdir -p public/models && cp ~/devfs/cache/parakeet-ggml-webgpu/gguf/parakeet-tdt_ctc-110m-Q4_0.gguf public/models/
bun run build && bun run serve        # http://127.0.0.1:8795/
```

`public/app.js`, `public/parakeet-web/` and `public/models/` are made by those commands.
