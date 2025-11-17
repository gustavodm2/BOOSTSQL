-- SQLBoost Database Tables Creation Script
-- Run this script in PostgreSQL to create the boostsql database tables
-- Database name: boostsql

-- Create the database (run this first, or create manually)
-- CREATE DATABASE boostsql;

-- Connect to the boostsql database before running the rest of this script

-- Create tables
CREATE TABLE categorias (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL,
    descricao TEXT
);

CREATE TABLE fornecedores (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    nome_contato VARCHAR(100),
    cidade VARCHAR(50),
    pais VARCHAR(50),
    telefone VARCHAR(20),
    email VARCHAR(100)
);

CREATE TABLE produtos (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    categoria VARCHAR(50),
    preco DECIMAL(10,2) NOT NULL,
    estoque INTEGER DEFAULT 0,
    fornecedor_id INTEGER REFERENCES fornecedores(id),
    avaliacao DECIMAL(3,2) DEFAULT 0.0,
    descricao TEXT,
    categoria_id INTEGER REFERENCES categorias(id)
);

CREATE TABLE clientes (
    id SERIAL PRIMARY KEY,
    nome_empresa VARCHAR(100),
    nome_contato VARCHAR(100) NOT NULL,
    cidade VARCHAR(50),
    pais VARCHAR(50),
    telefone VARCHAR(20),
    email VARCHAR(100) UNIQUE
);

CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    idade INTEGER,
    cidade VARCHAR(50),
    pais VARCHAR(50),
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'ativo',
    reputacao INTEGER DEFAULT 0
);

CREATE TABLE pedidos (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    cliente_id INTEGER REFERENCES clientes(id),
    data_pedido TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pendente',
    valor_total DECIMAL(10,2) DEFAULT 0.0
);

CREATE TABLE itens_pedido (
    id SERIAL PRIMARY KEY,
    pedido_id INTEGER REFERENCES pedidos(id),
    produto_id INTEGER REFERENCES produtos(id),
    quantidade INTEGER NOT NULL,
    preco_unitario DECIMAL(10,2) NOT NULL
);

CREATE TABLE pagamentos (
    id SERIAL PRIMARY KEY,
    pedido_id INTEGER REFERENCES pedidos(id),
    valor DECIMAL(10,2) NOT NULL,
    data_pagamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metodo VARCHAR(20),
    status VARCHAR(20) DEFAULT 'concluido'
);

CREATE TABLE sessoes_usuario (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    hora_login TIMESTAMP NOT NULL,
    hora_logout TIMESTAMP,
    endereco_ip INET
);

CREATE TABLE avaliacoes_produto (
    id SERIAL PRIMARY KEY,
    produto_id INTEGER REFERENCES produtos(id),
    usuario_id INTEGER REFERENCES usuarios(id),
    nota INTEGER CHECK (nota >= 1 AND nota <= 5),
    comentario TEXT,
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE departamentos (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL,
    orcamento DECIMAL(12,2)
);

CREATE TABLE funcionarios (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    departamento_id INTEGER REFERENCES departamentos(id),
    salario DECIMAL(10,2),
    data_contratacao DATE,
    email VARCHAR(100) UNIQUE
);

CREATE TABLE estoque (
    id SERIAL PRIMARY KEY,
    produto_id INTEGER REFERENCES produtos(id),
    local_armazem VARCHAR(50),
    quantidade INTEGER DEFAULT 0,
    ultima_atualizacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE vendas (
    id SERIAL PRIMARY KEY,
    produto_id INTEGER REFERENCES produtos(id),
    funcionario_id INTEGER REFERENCES funcionarios(id),
    quantidade INTEGER,
    data_venda TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valor_total DECIMAL(10,2)
);

CREATE TABLE transacoes (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    valor DECIMAL(10,2),
    data_transacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tipo VARCHAR(20),
    status VARCHAR(20)
);

CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    acao VARCHAR(100),
    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    endereco_ip INET,
    detalhes TEXT
);

CREATE TABLE perfis (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    bio TEXT,
    url_avatar VARCHAR(255),
    preferencias JSONB
);

CREATE TABLE envios (
    id SERIAL PRIMARY KEY,
    pedido_id INTEGER REFERENCES pedidos(id),
    codigo_rastreio VARCHAR(50),
    transportadora VARCHAR(50),
    status VARCHAR(20),
    data_envio TIMESTAMP,
    data_entrega TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_usuarios_email ON usuarios(email);
CREATE INDEX idx_usuarios_city ON usuarios(cidade);
CREATE INDEX idx_pedidos_usuario_id ON pedidos(usuario_id);
CREATE INDEX idx_pedidos_date ON pedidos(data_pedido);
CREATE INDEX idx_produtos_categoria ON produtos(categoria);
CREATE INDEX idx_produtos_preco ON produtos(preco);
CREATE INDEX idx_itens_pedido_pedido_id ON itens_pedido(pedido_id);
CREATE INDEX idx_pagamentos_pedido_id ON pagamentos(pedido_id);
CREATE INDEX idx_sessoes_usuario_usuario_id ON sessoes_usuario(usuario_id);
CREATE INDEX idx_avaliacoes_produto_produto_id ON avaliacoes_produto(produto_id);
CREATE INDEX idx_logs_usuario_id ON logs(usuario_id);
CREATE INDEX idx_transacoes_usuario_id ON transacoes(usuario_id);

COMMIT;

-- Summary: This creates all the database tables with proper relationships and indexes.
-- Run this SQL file first, then run the Python script to insert 100,000+ rows of data.