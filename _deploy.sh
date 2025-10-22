#!/bin/bash

MESSAGE="$1"

# Default message
if [ -z "$MESSAGE" ]; then
  MESSAGE="Update site"
fi

echo "🟢 Git add..."
git add .

echo "🟡 Git commit..."
git commit -m "$MESSAGE"

echo "🔵 Git push to GitHub..."
git push origin master

echo "✅ Deployed successfully!"
