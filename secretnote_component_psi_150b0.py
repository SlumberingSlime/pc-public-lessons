# ref: https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.6.1b0/component/comp_guide
# warning: the psi version on this guide is outdated!
import json

from secretflow.component.entry import comp_eval
from secretflow.spec.extend.cluster_pb2 import (
    SFClusterConfig,
    SFClusterDesc,
)
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    TableSchema,
    IndividualTable,
    StorageConfig,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
import click


@click.command()
@click.argument("party", type=str)
def run(party: str):
    desc = SFClusterDesc(
        parties=["alice", "bob"],
        devices=[
            SFClusterDesc.DeviceDesc(
                name="spu",
                type="spu",
                parties=["alice", "bob"],
                config=json.dumps(
                    {
                        "runtime_config": {"protocol": "REF2K", "field": "FM64"},
                        "link_desc": {
                            "connect_retry_times": 60,
                            "connect_retry_interval_ms": 1000,
                            "brpc_channel_protocol": "http",
                            "brpc_channel_connection_type": "pooled",
                            "recv_timeout_ms": 1200 * 1000,
                            "http_timeout_ms": 1200 * 1000,
                        },
                    }
                ),
            ),
            SFClusterDesc.DeviceDesc(
                name="heu",
                type="heu",
                parties=[],
                config=json.dumps(
                    {
                        "mode": "PHEU",
                        "schema": "paillier",
                        "key_size": 2048,
                    }
                ),
            ),
        ],
    )

    sf_cluster_config = SFClusterConfig(
        desc=desc,
        public_config=SFClusterConfig.PublicConfig(
            ray_fed_config=SFClusterConfig.RayFedConfig(
                parties=["alice", "bob"],
                addresses=[
                    "127.0.0.1:61041",
                    "127.0.0.1:61042",
                ],
            ),
            spu_configs=[
                SFClusterConfig.SPUConfig(
                    name="spu",
                    parties=["alice", "bob"],
                    addresses=[
                        "127.0.0.1:61045",
                        "127.0.0.1:61046",
                    ],
                )
            ],
        ),
        private_config=SFClusterConfig.PrivateConfig(
            self_party=party,
            ray_head_addr="local",  # local means setup a Ray cluster instead connecting to an existed one.
        ),
    )

    # check https://www.secretflow.org.cn/docs/spec/latest/zh-Hans/intro#nodeevalparam for details.
    sf_node_eval_param = NodeEvalParam(
        domain="data_prep",
        name="psi",
        version="0.0.4",
        attr_paths=[
            "protocol",
            "disable_alignment",
            "ecdh_curve",
	    "left_side",
            "input/receiver_input/key",
            "input/sender_input/key",
        ],
        attrs=[
            Attribute(s="PROTOCOL_ECDH"),
            Attribute(b=True),
            Attribute(s="CURVE_FOURQ"),
	    Attribute(ss=["alice"]),
            Attribute(ss=["id1"]),
            Attribute(ss=["id2"]),
        ],
        inputs=[
            DistData(
                name="receiver_input",
                type="sf.table.individual",
                data_refs=[
                    DistData.DataRef(uri="input.csv", party="alice", format="csv"),
                ],
            ),
            DistData(
                name="sender_input",
                type="sf.table.individual",
                data_refs=[
                    DistData.DataRef(uri="input.csv", party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[
            "output.csv",
        ],
    )

    sf_node_eval_param.inputs[0].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                id_types=["str"],
                ids=["id1"],
            ),
            line_count=-1,
        ),
    )

    sf_node_eval_param.inputs[1].meta.Pack(
        IndividualTable(
            schema=TableSchema(
                id_types=["str"],
                ids=["id2"],
            ),
            line_count=-1,
        ),
    )

    storage_config = StorageConfig(
        type="local_fs",
        local_fs=StorageConfig.LocalFSConfig(wd=f"/tmp/{party}"),
    )

    res = comp_eval(sf_node_eval_param, storage_config, sf_cluster_config)

    print(f'Node eval res is \n{res}')


if __name__ == "__main__":
    run()
