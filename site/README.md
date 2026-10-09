# Ratio project page

This folder is a static page that describes Ratio; Ratio itself runs offline, on the reviewer's own computer. The page has no scripts, cookies, analytics, forms or external resources. Its host, Vercel, keeps standard request logs.

Vercel publishes every file in this folder except this README, which `.vercelignore` keeps off the site.

To deploy it on Vercel:

0. Commit `site/` to `main`, because Vercel deploys production from `main`. Make the repository public before sharing the link, because the page links to it.
1. Import the GitHub repository as a new project, and set **Root Directory** to `site`.
2. Set **Framework Preset** to **Other** and leave the build command empty: there is no build step.
3. Deploy. `vercel.json` adds the security headers, including a strict Content-Security-Policy, to every path.
4. Open the production domain listed under the project's **Settings > Domains**, not the per-deployment URL, which asks for a Vercel login. Then check the deployment:

   ```bash
   curl -sI https://<domain>/ | grep -iE 'content-security-policy|referrer-policy|x-content-type-options|permissions-policy'
   curl -s -o /dev/null -w '%{http_code}\n' https://<domain>/README.md
   ```

   The first command must print all four headers; if it prints none, the Root Directory is wrong and `vercel.json` was not read. The second must print `404`; if it prints `200`, this README is public, so move it out of `site/` and deploy again.
