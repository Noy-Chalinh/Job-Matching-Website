from whitenoise.storage import CompressedManifestStaticFilesStorage


class ForgivingManifestStaticFilesStorage(CompressedManifestStaticFilesStorage):
    """WhiteNoise's compressed, content-hashed storage, minus the hard failure
    when collectstatic hasn't been run (local `runserver` with DEBUG off, or
    a deploy whose build command skips build.sh).

    The stock storage raises "Missing staticfiles manifest entry" from every
    {% static %} tag in that case, which 500s the whole page. This falls
    back to the plain, unhashed path instead, which WhiteNoise still serves
    straight from the app's static/ directory (WHITENOISE_USE_FINDERS in
    settings). After collectstatic, hashed cache-forever URLs are used as
    normal.
    """

    def stored_name(self, name):
        try:
            return super().stored_name(name)
        except ValueError:
            return name
