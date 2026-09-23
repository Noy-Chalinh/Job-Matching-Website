from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enables Postgres's `vector` extension so JobEmbedding.vector (added in
    the next migration) can use the `vector(384)` column type. VectorExtension
    (pgvector.django) wraps django.contrib.postgres's CreateExtension, which
    already no-ops on non-Postgres backends (checks connection.vendor
    internally) - safe to run against SQLite local dev.

    Requires 'django.contrib.postgres' in INSTALLED_APPS (see
    config/settings.py) and that the Supabase project has the vector
    extension enabled for this database (Database -> Extensions in the
    Supabase dashboard) - the app's connection role may not have
    CREATE EXTENSION privilege itself.
    """

    dependencies = [
        ('jobs', '0003_skillembedding'),
    ]

    operations = [
        VectorExtension(),
    ]
