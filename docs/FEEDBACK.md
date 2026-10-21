# Feedback on Nebius Token Factory, Sandboxes and NVIDIA tools

A running log written while building. Each item says what happened, why it mattered, and a suggested fix.

## Token Factory Sandboxes and contree-sdk

1. **Two names and two endpoints for one product.** The SDK and MCP server call the product ConTree and default to contree.dev, while the hackathon and the Token Factory docs call it Sandboxes at api.tokenfactory.nebius.com/sandboxes. I had to read the SDK source (`ContreeEndpoint.TOKEN_FACTORY_SANDBOXES`, the `IAMAuth` defaults) to be sure which endpoint and auth scheme applies. *Suggestion:* one quickstart page for Token Factory users that states the endpoint, the auth headers and the SDK constructor near the top.

2. **The Project header is easy to miss.** `IAMAuth` sends both a bearer token and a `Project` header, and sandbox access must be enabled for that project. A missing project ID fails at request time rather than when the client is built. *Suggestion:* validate that a project ID is present when constructing the client, with an error that says where to find it.

3. **Inconsistent naming for elapsed time.** `ContreeResult` exposes `elapsed_time`, while the image object exposes the same value as `elapsed`. I first wrote `out.elapsed_time` on the image and it failed. *Suggestion:* one name, or an alias for the other.

4. **Output truncation is silent unless you know to check.** `default_truncate_output_at` is 65,535 bytes, so returning results on stdout breaks for anything beyond a small payload. Reading a file back from the resulting image works well, but only with `disposable=False`. *Suggestion:* document "write a file, keep the image, read it back" as the standard way to return data, and warn when output was truncated.

5. **Upload paths.** It was unclear from the docs whether uploading to a path whose parent directory doesn't exist creates the directory. I create `/workspace` in the setup step to be safe. *Suggestion:* state the behaviour in the `files=` docstring.

6. **What went well:** the checkpoint-and-branch model is exactly right for experiment grids. Setup runs once, every branch is isolated, and each result reports its own compute time and cost, which made per-trial cost accounting straightforward. It also enabled something I haven't seen pitched before: using isolation as a scientific guarantee (the research checkpoint physically doesn't contain the held-out data).

## Token Factory inference and Nemotron

<!-- Add during live testing: model ID discovery, JSON mode reliability per tier, whether reasoning traces appear in content, latency by tier, rate limits. -->

## Tavily

<!-- Add during live testing. -->
