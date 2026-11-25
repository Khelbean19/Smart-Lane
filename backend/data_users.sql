INSERT INTO users (
        full_name,
        username,
        email,
        phone,
        password_hash,
        role,
        status
    )
VALUES (
        'Zohran Mamdani',
        'zohranmamdani',
        'zohran.mamdani@smartlane.com',
        '639123456789',
        '$argon2id$v=19$m=65536,t=3,p=4$6Gm4/NXrUnot0onDWvdJlQ$IMfx6VW0BVY3YIt+OWtYBgLW/JAFiHX3oJQXQ8bx2Xs', -- password-admin123
        'ADMINISTRATOR',
        'ACTIVE'
    ),
    (
        'James Talarico',
        'jamestalarico',
        'james.talarico@smartlane.com',
        '639987654321',
        '$argon2id$v=19$m=65536,t=3,p=4$6Gm4/NXrUnot0onDWvdJlQ$IMfx6VW0BVY3YIt+OWtYBgLW/JAFiHX3oJQXQ8bx2Xs', -- password-admin123
        'ADMINISTRATOR',
        'ACTIVE'
    ),
    (
        'Marco Rubio',
        'marcorubio',
        'marco.rubio@smartlane.com',
        '639671234567',
        '$argon2id$v=19$m=65536,t=3,p=4$vnFo740ZnpshKVreTnq6xg$cA00dQ1nCUOli4HguZp+DhWkhPWfxE2l6KokCKxmtd4', -- password-enforcer123
        'ENFORCER',
        'ACTIVE'
    ),
    (
        'Pete Hegseth',
        'petehegseth',
        'pete.hegseth@smartlane.com',
        '639911245836',
        '$argon2id$v=19$m=65536,t=3,p=4$vnFo740ZnpshKVreTnq6xg$cA00dQ1nCUOli4HguZp+DhWkhPWfxE2l6KokCKxmtd4', -- password-enforcer123
        'ENFORCER',
        'ACTIVE'
    ),
    (
        'JD Vance',
        'jdvance',
        'jd.vance@smartlane.com',
        '639674562567',
        '$argon2id$v=19$m=65536,t=3,p=4$vnFo740ZnpshKVreTnq6xg$cA00dQ1nCUOli4HguZp+DhWkhPWfxE2l6KokCKxmtd4', -- password-enforcer123
        'ENFORCER',
        'ACTIVE'
    );