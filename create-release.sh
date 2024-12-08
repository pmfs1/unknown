#! /bin/bash
git tag "$(git log -1 --pretty=format:"%s")" && git push --force origin HEAD:trunk && git push origin "$(git log -1 --pretty=format:"%s")"