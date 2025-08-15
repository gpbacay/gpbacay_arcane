# A.R.C.A.N.E. Deployment Guide
Augmented Reconstruction of Consciousness through Artificial Neural Evolution

## Overview
This guide covers deploying the A.R.C.A.N.E. neuromimetic language foundation model web application to various cloud platforms.

## Prerequisites
1. **Trained Model**: Run `python train_neuromimetic_lm.py` to train your neuromimetic language model
2. **Model Files**: Ensure you have:
   - `Models/neuromimetic_lm.keras` (trained model)
   - `Models/neuromimetic_lm_tokenizer.pickle` (tokenizer)

## Deployment Options

### 1. Heroku Deployment

**Step 1**: Install Heroku CLI and login
```bash
# Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login
```

**Step 2**: Create Heroku app
```bash
# Create new Heroku app
heroku create arcane-neuromimetic

# Add Python buildpack
heroku buildpacks:set heroku/python

# Set environment variables
heroku config:set SECRET_KEY="your-secret-key-here"
heroku config:set DEBUG=False
```

**Step 3**: Deploy
```bash
# Add Heroku remote
git add .
git commit -m "Deploy A.R.C.A.N.E. neuromimetic language model"
git push heroku main

# Open the deployed app
heroku open
```

### 2. Railway Deployment

**Step 1**: Install Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login
```

**Step 2**: Deploy
```bash
# Initialize Railway project
railway init

# Set environment variables
railway add SECRET_KEY
railway add DEBUG=False

# Deploy
railway up
```

### 3. Render Deployment

**Step 1**: Create account at [render.com](https://render.com)

**Step 2**: Connect GitHub repository

**Step 3**: Create new Web Service with these settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `cd arcane_project && gunicorn arcane_project.wsgi:application --bind 0.0.0.0:$PORT`

**Step 4**: Add environment variables:
- `SECRET_KEY`: Your secret key
- `DEBUG`: `False`
- `PYTHON_VERSION`: `3.11.0`

### 4. Vercel Deployment

**Step 1**: Install Vercel CLI
```bash
npm install -g vercel
```

**Step 2**: Configure `vercel.json`
```json
{
  "builds": [
    {
      "src": "arcane_project/wsgi.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "arcane_project/wsgi.py"
    }
  ]
}
```

**Step 3**: Deploy
```bash
vercel --prod
```

## Environment Variables

Set these environment variables on your deployment platform:

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | `your-secret-key-here` |
| `DEBUG` | Debug mode | `False` |
| `CUSTOM_DOMAIN` | Your custom domain | `arcane.yourdomain.com` |
| `DATABASE_URL` | Database URL (optional) | `postgres://user:pass@host:port/db` |

## Model Files

**Important**: Due to file size limitations, you'll need to:

1. **Upload model files separately** to your deployment platform's storage
2. **Use cloud storage** (AWS S3, Google Cloud Storage) to host model files
3. **Update settings** to point to your model file locations

Example cloud storage setup:
```python
# In settings.py
MODEL_SETTINGS = {
    'MODEL_PATH': 'https://your-bucket.s3.amazonaws.com/neuromimetic_lm.keras',
    'TOKENIZER_PATH': 'https://your-bucket.s3.amazonaws.com/neuromimetic_lm_tokenizer.pickle',
}
```

## Post-Deployment

1. **Test the deployment**:
   - Visit your deployed URL
   - Try generating text with different seed phrases
   - Check model status indicator

2. **Monitor performance**:
   - Check application logs
   - Monitor response times
   - Verify model loading successfully

3. **Custom domain** (optional):
   - Configure DNS settings
   - Add SSL certificate
   - Update `ALLOWED_HOSTS` in settings

## Troubleshooting

### Common Issues:

1. **Model not loading**:
   - Check model file paths in settings
   - Verify model files are accessible
   - Check application logs for errors

2. **Slow response times**:
   - Model loading on first request (cold start)
   - Consider model optimization
   - Use application warm-up strategies

3. **Memory issues**:
   - TensorFlow models require significant memory
   - Choose deployment plans with adequate RAM (1GB+ recommended)
   - Configure TensorFlow memory growth settings

### Debug Commands:
```bash
# Check deployment logs
heroku logs --tail  # Heroku
railway logs        # Railway

# Test locally
python arcane_project/manage.py runserver

# Check model loading
python arcane_project/manage.py shell
>>> from recognizer.views import model_loaded_successfully
>>> print(model_loaded_successfully)
```

## Production Optimization

1. **Model optimization**:
   - Use TensorFlow Lite for smaller model size
   - Quantization for faster inference
   - Model pruning techniques

2. **Caching**:
   - Implement Redis for model caching
   - Cache generated text for common seeds
   - Use CDN for static files

3. **Monitoring**:
   - Set up application monitoring (Sentry)
   - Track model performance metrics
   - Monitor resource usage

## Security

1. **Environment variables**:
   - Never commit secret keys to git
   - Use platform-specific secret management
   - Rotate keys regularly

2. **Rate limiting**:
   - Implement request rate limiting
   - Prevent abuse of text generation API
   - Monitor unusual usage patterns

3. **Input validation**:
   - Sanitize user input
   - Limit text generation parameters
   - Implement CSRF protection

## Success Criteria

Your deployment is successful when:
- ✅ Website loads without errors
- ✅ Model status shows "Model Ready"
- ✅ Text generation works with sample inputs
- ✅ All neuromimetic features are functional
- ✅ Response times are acceptable (< 10 seconds for generation)

## Support

For deployment issues:
1. Check the application logs
2. Verify all environment variables are set
3. Ensure model files are accessible
4. Test locally first before deploying

Your A.R.C.A.N.E. neuromimetic language foundation model is now ready for the world! 🧠🚀
