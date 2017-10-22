var path = require('path')

module.exports = {
    entry: './entry.js', 

    output: {
        path: path.join(__dirname, 'dist'),
        filename: 'bundle.js'
    },
    
    devtool: 'source-map',
    
    devServer: {
        contentBase: 'dist'
    },
    node: {
      fs: 'empty'
    },
    module: {
      loaders: [
        { test: /\.(glsl)$/, loader: 'url-loader'}
      ]
    }
  };