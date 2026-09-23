import django.db.models.deletion
from django.db import migrations, models
from pgvector.django import VectorField


def create_job_embedding_hnsw_index(apps, schema_editor):
    """pgvector.django.HnswIndex generates `CREATE INDEX ... USING hnsw
    (...)`, which is Postgres-only syntax - SQLite's CREATE INDEX has no
    USING clause and would raise a syntax error. Declaring HnswIndex in
    JobEmbedding.Meta.indexes would apply it unconditionally on whichever
    backend `migrate` runs against (SQLite for local dev, Postgres for
    prod), so it's created here instead via raw SQL, gated explicitly on
    connection.vendor - mirrors this app's existing graceful-degradation
    convention (see jobs/models.py's JobEmbedding docstring,
    jobs/services/matcher.py's use_semantic fallback).
    """
    if schema_editor.connection.vendor != 'postgresql':
        return
    schema_editor.execute(
        'CREATE INDEX job_embeddings_vector_hnsw ON job_embeddings '
        'USING hnsw (vector vector_cosine_ops)'
    )


def drop_job_embedding_hnsw_index(apps, schema_editor):
    if schema_editor.connection.vendor != 'postgresql':
        return
    schema_editor.execute('DROP INDEX IF EXISTS job_embeddings_vector_hnsw')


class Migration(migrations.Migration):

    dependencies = [
        ('jobs', '0004_vector_extension'),
    ]

    operations = [
        migrations.CreateModel(
            name='JobEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vector', VectorField(dimensions=384)),
                ('model_name', models.CharField(default='BAAI/bge-small-en-v1.5', max_length=100)),
                ('dim', models.PositiveSmallIntegerField(default=384)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('job', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='embedding', to='jobs.job')),
            ],
            options={
                'db_table': 'job_embeddings',
            },
        ),
        migrations.RunPython(create_job_embedding_hnsw_index, drop_job_embedding_hnsw_index),
    ]
