// SPDX-License-Identifier: Apache 
pragma solidity ^0.8.0;

contract SupervisedWallet {
    uint private numTransactions;

	address private supervisor;
    address private supervisee;
	uint public funds;

    struct Transaction {
        address payable recipient;
        uint amount;
    }

    mapping(uint => Transaction) private transactions;

    event TransactionCreated(uint transactionId);
    event TransactionApproved(uint transactionId);
    event TransactionDenied(uint transactionId);

	constructor(address _supervisee) payable {
		supervisor = msg.sender;
		funds = msg.value;
        supervisee = _supervisee;
	}

	modifier onlySupervisor {
		_;
		require(msg.sender == supervisor, "Only the supervisor may call this function");
	}

    modifier onlySupervisee {
        _;
        require(msg.sender == supervisor, "Only the supervisee may call this function");
    }

	function deposit() external payable onlySupervisor {
		funds += msg.value;
	}

    function pay(address payable _recipient, uint _amount) external onlySupervisee {
        Transaction memory t = transactions[numTransactions];
        t.recipient = _recipient;
        t.amount = _amount;
        emit TransactionCreated(numTransactions++);
    }

    function approveTransaction(uint transactionId) external onlySupervisor {
        Transaction memory t = transactions[transactionId];
        t.recipient.transfer(t.amount);
        delete transactions[transactionId];
        emit TransactionApproved(transactionId);
    }

    function denyTransaction(uint transactionId) external onlySupervisor {
        delete transactions[transactionId];
        emit TransactionDenied(transactionId);
    }
}
