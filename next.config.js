const isProd = process.env.NODE_ENV === "production";

module.exports = {
  assetPrefix: isProd ? "/recycle-bot" : "",
  basePath: isProd ? "/recycle-bot" : "",
  env: {
    BASE_URL: isProd ? "/recycle-bot" : "",
  },
};
