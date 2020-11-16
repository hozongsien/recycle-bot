const isProd = process.env.NODE_ENV === 'production'

module.exports = {
  env: {
    BACKEND_URL: isProd ? "/recycle-bot" : "",
  },
};
