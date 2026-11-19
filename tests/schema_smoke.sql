\set ON_ERROR_STOP on
BEGIN;

DO $$
DECLARE
    policy_id BIGINT;
    test_embedding vector := array_fill(0.01::real, ARRAY[1536])::vector;
BEGIN
    IF (SELECT count(*) FROM users WHERE username IN ('alice', 'bob', 'carol', 'guest')) <> 4 THEN
        RAISE EXCEPTION 'Demo users missing';
    END IF;

    INSERT INTO documents(source, title, department, visibility, effective_from, updated_on, content_hash)
    VALUES ('schema-smoke.md', '__schema_smoke__', 'hr', 'restricted', CURRENT_DATE, CURRENT_DATE, repeat('a', 64))
    RETURNING id INTO policy_id;

    BEGIN
        INSERT INTO documents(source, title, department, visibility, effective_from, updated_on, content_hash)
        VALUES ('duplicate.md', '__schema_smoke_duplicate__', 'hr', 'restricted', CURRENT_DATE, CURRENT_DATE, repeat('a', 64));
        RAISE EXCEPTION 'Duplicate hash accepted';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    BEGIN
        INSERT INTO documents(source, title, department, visibility, effective_from, updated_on, content_hash)
        VALUES ('duplicate-version.md', '__schema_smoke__', 'hr', 'restricted', CURRENT_DATE, CURRENT_DATE, repeat('b', 64));
        RAISE EXCEPTION 'Duplicate title/version accepted';
    EXCEPTION WHEN unique_violation THEN NULL;
    END;

    BEGIN
        UPDATE documents SET visibility = 'unknown' WHERE id = policy_id;
        RAISE EXCEPTION 'Invalid visibility accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;

    BEGIN
        UPDATE documents SET version = 0 WHERE id = policy_id;
        RAISE EXCEPTION 'Invalid version accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;

    BEGIN
        UPDATE documents SET status = 'unknown' WHERE id = policy_id;
        RAISE EXCEPTION 'Invalid status accepted';
    EXCEPTION WHEN check_violation THEN NULL;
    END;

    BEGIN
        INSERT INTO chunks(document_id, section, content, embedding)
        VALUES (policy_id, 'Leave', 'Employees receive annual leave.', '[1,2,3]'::vector);
        RAISE EXCEPTION 'Invalid embedding dimension accepted';
    EXCEPTION WHEN data_exception THEN NULL;
    END;

    INSERT INTO chunks(document_id, section, content, embedding)
    VALUES (policy_id, 'Leave', 'Employees receive annual leave.', test_embedding);

    IF NOT EXISTS (SELECT 1 FROM chunks WHERE document_id = policy_id
                   AND search_text @@ plainto_tsquery('english', 'annual leave')) THEN
        RAISE EXCEPTION 'Generated full-text index failed';
    END IF;

    INSERT INTO audit_events(username, question, answered, grounded, sources)
    VALUES ('bob', 'What is the leave policy?', TRUE, TRUE, '__schema_smoke__ v1');

    DELETE FROM documents WHERE id = policy_id;
    IF EXISTS (SELECT 1 FROM chunks WHERE document_id = policy_id) THEN
        RAISE EXCEPTION 'Chunk cascade failed';
    END IF;
    RAISE NOTICE 'Schema smoke checks passed';
END $$;

-- Leave no test policies, chunks, or audit events behind.
ROLLBACK;
